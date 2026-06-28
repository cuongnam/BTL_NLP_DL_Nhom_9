"""
train_trigger_model.py - Huấn luyện Run 01 (Rule-based) và Run 02 (BiLSTM-CRF)

Sửa so với phiên bản cũ:
  1. [Run 01] Đổi sang đánh giá F1 theo SPAN (seqeval) thay vì token-level,
     để kết quả so sánh được với Run 03/04.
  2. [Run 02] Thay SimpleCRF giả bằng torchcrf thực sự (có Viterbi decoding).
  3. [Run 02] Bổ sung seqeval để đánh giá F1 theo span.
  4. Thêm seed cố định để tái lập kết quả.

Yêu cầu cài thêm:
  pip install torchcrf seqeval
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader, Dataset
from torchcrf import CRF          # pip install torchcrf
from seqeval.metrics import f1_score, precision_score, recall_score

# ─── Cấu hình chung ──────────────────────────────────────────────────────────
TRAIN_BIO_FILE = "./data/processed_bio/train_bio.json"
DEV_BIO_FILE   = "./data/processed_bio/dev_bio.json"
WANDB_PROJECT  = "BKEE_Event_Extraction_LREC2024"
SEED           = 42

LABEL_LIST    = ["O", "B-TRIGGER", "I-TRIGGER"]
LABEL_TO_IDX  = {l: i for i, l in enumerate(LABEL_LIST)}
IDX_TO_LABEL  = {i: l for i, l in enumerate(LABEL_LIST)}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 01 — Rule-Based Trigger (Exact Match từ điển)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_span_f1(pred_label_seqs, gold_label_seqs):
    """Tính P/R/F1 theo span dùng seqeval."""
    p = precision_score(gold_label_seqs, pred_label_seqs)
    r = recall_score(gold_label_seqs, pred_label_seqs)
    f = f1_score(gold_label_seqs, pred_label_seqs)
    return p, r, f


def run_01_rule_based():
    print("\n" + "=" * 60)
    print("KHỞI CHẠY RUN 01: RULE-BASED TRIGGER")
    print("=" * 60)

    wandb.init(
        project=WANDB_PROJECT,
        name="run_01_rule_based_trigger",
        config={"architecture": "Rule-Based", "method": "Exact Match Dictionary"},
    )

    # 1. Xây dựng từ điển trigger từ tập Train
    trigger_vocab = set()
    with open(TRAIN_BIO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for token, label in zip(item["tokens"], item["labels"]):
                if label != "O":
                    trigger_vocab.add(token.lower())

    print(f"  Từ điển trigger: {len(trigger_vocab)} từ")

    # 2. Dự đoán và đánh giá theo SPAN trên tập Dev
    all_preds = []
    all_golds = []

    with open(DEV_BIO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            tokens    = item["tokens"]
            gold_labels = item["labels"]

            # Dự đoán: token nào có trong vocab → B-TRIGGER
            pred_labels = []
            for token in tokens:
                if token.lower() in trigger_vocab:
                    pred_labels.append("B-TRIGGER")
                else:
                    pred_labels.append("O")

            all_preds.append(pred_labels)
            all_golds.append(gold_labels)

    precision, recall, f1 = compute_span_f1(all_preds, all_golds)

    print(f"\n--- KẾT QUẢ RUN 01 (Span-level seqeval) ---")
    print(f"  Precision : {precision * 100:.2f}%")
    print(f"  Recall    : {recall    * 100:.2f}%")
    print(f"  F1-score  : {f1        * 100:.2f}%")

    wandb.log({"precision": precision, "recall": recall, "f1_score": f1})
    wandb.finish()
    return f1


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 02 — BiLSTM + CRF thực sự (torchcrf)
# ═══════════════════════════════════════════════════════════════════════════════

class BKEETriggerDataset(Dataset):
    def __init__(self, file_path, word_to_idx):
        self.features = []
        self.labels   = []

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item   = json.loads(line.strip())
                w_ids  = [word_to_idx.get(w.lower(), word_to_idx["<UNK>"]) for w in item["tokens"]]
                l_ids  = [LABEL_TO_IDX.get(l, LABEL_TO_IDX["O"]) for l in item["labels"]]
                self.features.append(torch.tensor(w_ids, dtype=torch.long))
                self.labels.append(torch.tensor(l_ids, dtype=torch.long))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs   = nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = nn.utils.rnn.pad_sequence(labels,    batch_first=True, padding_value=0)
    mask = (padded_seqs != 0)  # True ở vị trí thực, False ở padding
    return padded_seqs, padded_labels, mask


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_tags, embed_dim=128, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.dropout   = nn.Dropout(dropout)
        self.lstm      = nn.LSTM(
            embed_dim, hidden_dim // 2,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=dropout,
        )
        self.fc  = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, batch_first=True)   # torchcrf — CRF thực sự

    def forward(self, x, mask):
        """Trả về negative log-likelihood (dùng làm loss khi training)."""
        embeds    = self.dropout(self.embedding(x))
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(self.dropout(lstm_out))
        return emissions

    def loss(self, emissions, tags, mask):
        # CRF forward: trả về log-likelihood → negate để thành loss
        return -self.crf(emissions, tags, mask=mask, reduction="mean")

    def decode(self, emissions, mask):
        """Viterbi decoding — trả về list of list of int."""
        return self.crf.decode(emissions, mask=mask)


def evaluate_bilstm(model, loader, device):
    """Đánh giá F1 theo span dùng seqeval."""
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for seqs, labels, mask in loader:
            seqs, mask = seqs.to(device), mask.to(device)
            emissions  = model(seqs, mask)
            pred_ids   = model.decode(emissions, mask)   # list of list

            # Chuyển từ id → nhãn chuỗi
            labels_np = labels.numpy()
            mask_np   = mask.cpu().numpy()

            for i, pred_seq in enumerate(pred_ids):
                length    = int(mask_np[i].sum())
                pred_lbls = [IDX_TO_LABEL[p] for p in pred_seq[:length]]
                gold_lbls = [IDX_TO_LABEL[labels_np[i][j]] for j in range(length)]
                all_preds.append(pred_lbls)
                all_golds.append(gold_lbls)

    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f = f1_score(all_golds, all_preds)
    return p, r, f


def run_02_bilstm_crf():
    print("\n" + "=" * 60)
    print("KHỞI CHẠY RUN 02: BiLSTM-CRF TRIGGER")
    print("=" * 60)

    set_seed()

    config = {
        "architecture": "BiLSTM-CRF",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "embed_dim": 128,
        "hidden_dim": 256,
        "dropout": 0.3,
        "optimizer": "Adam",
    }

    wandb.init(
        project=WANDB_PROJECT,
        name="run_02_bilstm_crf_trigger",
        config=config,
    )

    # Xây dựng từ vựng từ tập Train
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    with open(TRAIN_BIO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            for w in json.loads(line.strip())["tokens"]:
                if w.lower() not in word_to_idx:
                    word_to_idx[w.lower()] = len(word_to_idx)

    print(f"  Kích thước từ vựng: {len(word_to_idx)}")

    train_set = BKEETriggerDataset(TRAIN_BIO_FILE, word_to_idx)
    dev_set   = BKEETriggerDataset(DEV_BIO_FILE,   word_to_idx)

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,  collate_fn=collate_fn)
    dev_loader   = DataLoader(dev_set,   batch_size=config["batch_size"], shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Thiết bị: {device}")

    model     = BiLSTM_CRF(len(word_to_idx), len(LABEL_LIST),
                            embed_dim=config["embed_dim"],
                            hidden_dim=config["hidden_dim"],
                            dropout=config["dropout"]).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=2, factor=0.5)

    best_f1 = 0.0

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0.0

        for seqs, labels, mask in train_loader:
            seqs, labels, mask = seqs.to(device), labels.to(device), mask.to(device)

            optimizer.zero_grad()
            emissions = model(seqs, mask)
            loss      = model.loss(emissions, labels, mask)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        # Đánh giá trên Dev
        p, r, f1 = evaluate_bilstm(model, dev_loader, device)
        scheduler.step(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), "./saved_models/run_02_best.pt")

        wandb.log({
            "epoch": epoch + 1,
            "train/loss": avg_loss,
            "eval/precision": p,
            "eval/recall": r,
            "eval/f1": f1,
            "eval/best_f1": best_f1,
        })

        print(f"  Epoch {epoch+1:02d}/{config['epochs']} | "
              f"Loss: {avg_loss:.4f} | P: {p*100:.1f}% | R: {r*100:.1f}% | F1: {f1*100:.1f}% | Best: {best_f1*100:.1f}%")

    print(f"\n  Best F1 trên Dev: {best_f1 * 100:.2f}%")
    wandb.finish()
    return best_f1


if __name__ == "__main__":
    os.makedirs("./saved_models", exist_ok=True)
    os.makedirs("./data/processed_bio", exist_ok=True)

    run_01_rule_based()
    run_02_bilstm_crf()