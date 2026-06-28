"""
train_phobert_model.py - Huấn luyện Run 03 (PhoBERT thuần) và Run 04 (PhoBERT Joint)

Sửa so với phiên bản cũ:
  1. [Run 04] Label list được xây dựng CỐ ĐỊNH (không dùng sorted(set) động)
     để đảm bảo thứ tự nhất quán giữa các lần chạy.
  2. Tận dụng trường `pieces` và `token_lens` có sẵn trong BKEE thay vì
     re-tokenize bằng tokenizer (nhanh hơn và chính xác hơn).
  3. Bổ sung early stopping thủ công để tránh overfitting.
  4. Lưu label_list ra file JSON để dùng lại khi inference.
  5. Thêm seed cố định.

Yêu cầu:
  pip install transformers torch evaluate seqeval wandb
"""

import json
import os
import random

import numpy as np
import torch
import wandb
import evaluate
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)

# ─── Cấu hình ────────────────────────────────────────────────────────────────
MODEL_NAME    = "vinai/phobert-base-v2"
WANDB_PROJECT = "BKEE_Event_Extraction_LREC2024"
SEED          = 42
MAX_LEN       = 256   # tăng lên 256 vì BKEE có câu dài

TRAIN_BIO_FILE   = "./data/processed_bio/train_bio.json"
DEV_BIO_FILE     = "./data/processed_bio/dev_bio.json"
TRAIN_JOINT_FILE = "./data/processed_joint/train_joint.json"
DEV_JOINT_FILE   = "./data/processed_joint/dev_joint.json"


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ─── Xây dựng label list CỐ ĐỊNH cho Run 04 ─────────────────────────────────

def build_joint_label_list(train_joint_path):
    """
    Duyệt toàn bộ train để thu thập nhãn, sau đó sắp xếp theo quy tắc cố định:
      - "O" luôn ở index 0
      - B-* trước I-* cùng loại
      - Các loại sự kiện sort theo alphabet
    """
    label_set = set()
    with open(train_joint_path, "r", encoding="utf-8") as f:
        for line in f:
            for lbl in json.loads(line.strip())["labels"]:
                label_set.add(lbl)

    label_set.discard("O")

    # Tách B- và I-
    b_labels = sorted([l for l in label_set if l.startswith("B-")])
    i_labels = sorted([l for l in label_set if l.startswith("I-")])

    # Ghép: O trước, rồi cặp B-X / I-X theo alphabet
    event_types = sorted(set(l[2:] for l in b_labels))
    ordered = ["O"]
    for et in event_types:
        if f"B-{et}" in label_set:
            ordered.append(f"B-{et}")
        if f"I-{et}" in label_set:
            ordered.append(f"I-{et}")

    print(f"  Joint label list ({len(ordered)} nhãn): {ordered[:8]}{'...' if len(ordered) > 8 else ''}")
    return ordered


# ─── Dataset dùng `pieces` và `token_lens` có sẵn ────────────────────────────

class BKEEPhoBERTDataset(Dataset):
    """
    Tận dụng `pieces` và `token_lens` từ BKEE thay vì tokenize lại.

    token_lens[i] = số subword của tokens[i]
    Nhãn của subword đầu = nhãn của token gốc
    Nhãn của các subword sau = -100 (ignored by loss)
    """

    def __init__(self, file_path, tokenizer, label_to_idx, max_len=MAX_LEN):
        self.examples  = []
        self.tokenizer = tokenizer

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item       = json.loads(line)
                tokens     = item["tokens"]
                labels     = item["labels"]
                pieces     = item.get("pieces", [])
                token_lens = item.get("token_lens", [])

                # Nếu không có pieces, tokenize thủ công (fallback)
                if not pieces or not token_lens or len(token_lens) != len(tokens):
                    example = self._tokenize_fallback(tokens, labels, label_to_idx, max_len)
                else:
                    example = self._use_pieces(pieces, token_lens, labels, label_to_idx, max_len)

                if example is not None:
                    self.examples.append(example)

    def _use_pieces(self, pieces, token_lens, labels, label_to_idx, max_len):
        """Dùng pieces và token_lens có sẵn trong BKEE — chuẩn xác nhất."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        input_ids  = [cls_id]
        label_ids  = [-100]    # CLS token

        piece_idx = 0
        for token_idx, (n_pieces, label) in enumerate(zip(token_lens, labels)):
            token_pieces = pieces[piece_idx: piece_idx + n_pieces]
            piece_idx   += n_pieces

            piece_ids = self.tokenizer.convert_tokens_to_ids(token_pieces)
            if not piece_ids:
                continue

            input_ids.append(piece_ids[0])
            label_ids.append(label_to_idx.get(label, label_to_idx["O"]))

            for pid in piece_ids[1:]:
                input_ids.append(pid)
                label_ids.append(-100)   # subword phụ → ignore

        input_ids.append(sep_id)
        label_ids.append(-100)   # SEP token

        return self._pad_and_pack(input_ids, label_ids, max_len, pad_id)

    def _tokenize_fallback(self, tokens, labels, label_to_idx, max_len):
        """Fallback: tokenize lại bằng tokenizer (chậm hơn nhưng an toàn)."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        pad_id = self.tokenizer.pad_token_id

        input_ids = [cls_id]
        label_ids = [-100]

        for word, label in zip(tokens, labels):
            word_ids = self.tokenizer.encode(word, add_special_tokens=False)
            if not word_ids:
                continue
            input_ids.append(word_ids[0])
            label_ids.append(label_to_idx.get(label, label_to_idx["O"]))
            for wid in word_ids[1:]:
                input_ids.append(wid)
                label_ids.append(-100)

        input_ids.append(sep_id)
        label_ids.append(-100)

        return self._pad_and_pack(input_ids, label_ids, max_len, pad_id)

    def _pad_and_pack(self, input_ids, label_ids, max_len, pad_id):
        input_ids = input_ids[:max_len]
        label_ids = label_ids[:max_len]

        attn_mask = [1] * len(input_ids)
        pad_len   = max_len - len(input_ids)

        input_ids  += [pad_id] * pad_len
        label_ids  += [-100]   * pad_len
        attn_mask  += [0]      * pad_len

        return {
            "input_ids":      torch.tensor(input_ids,  dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask,  dtype=torch.long),
            "labels":         torch.tensor(label_ids,  dtype=torch.long),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─── Hàm tính metric ─────────────────────────────────────────────────────────

def make_compute_metrics(idx_to_label):
    seqeval = evaluate.load("seqeval")

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)

        true_preds = [
            [idx_to_label[pred] for pred, lbl in zip(prediction, label) if lbl != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [idx_to_label[lbl] for lbl in label if lbl != -100]
            for label in labels
        ]

        results = seqeval.compute(predictions=true_preds, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall":    results["overall_recall"],
            "f1":        results["overall_f1"],
        }

    return compute_metrics


def get_training_args(output_dir, run_name, epochs=5):
    return TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        report_to="wandb",
        run_name=run_name,
        seed=SEED,
        fp16=torch.cuda.is_available(),   # tăng tốc nếu có GPU
        dataloader_num_workers=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 03 — PhoBERT thuần (Trigger detection)
# ═══════════════════════════════════════════════════════════════════════════════

def run_03_phobert_pure():
    print("\n" + "=" * 60)
    print("KHỞI CHẠY RUN 03: PhoBERT Token Classification")
    print("=" * 60)

    set_seed()

    label_list   = ["O", "B-TRIGGER", "I-TRIGGER"]
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}

    # Lưu label list để dùng khi inference
    os.makedirs("./saved_models/run_03", exist_ok=True)
    with open("./saved_models/run_03/label_list.json", "w") as f:
        json.dump(label_list, f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=idx_to_label,
        label2id=label_to_idx,
    )

    print(f"  Số nhãn: {len(label_list)} — {label_list}")
    print(f"  Đang tải dataset...")

    train_dataset = BKEEPhoBERTDataset(TRAIN_BIO_FILE, tokenizer, label_to_idx)
    dev_dataset   = BKEEPhoBERTDataset(DEV_BIO_FILE,   tokenizer, label_to_idx)

    print(f"  Train: {len(train_dataset)} câu | Dev: {len(dev_dataset)} câu")

    trainer = Trainer(
        model=model,
        args=get_training_args("./saved_models/run_03", "run_03_phobert_trigger", epochs=5),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=make_compute_metrics(idx_to_label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n  [Run 03] Kết quả cuối: F1={metrics.get('eval_f1', 0)*100:.2f}%")
    wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 04 — PhoBERT Joint Learning (Trigger + Event Type đồng thời)
# ═══════════════════════════════════════════════════════════════════════════════

def run_04_phobert_joint():
    print("\n" + "=" * 60)
    print("KHỞI CHẠY RUN 04: PhoBERT Joint Trigger + Event Type")
    print("=" * 60)

    set_seed()

    # Label list CỐ ĐỊNH — không dùng sorted(set) ngẫu nhiên
    label_list   = build_joint_label_list(TRAIN_JOINT_FILE)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}

    # Lưu lại label list để inference sau này
    os.makedirs("./saved_models/run_04", exist_ok=True)
    with open("./saved_models/run_04/label_list.json", "w") as f:
        json.dump(label_list, f)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=idx_to_label,
        label2id=label_to_idx,
    )

    print(f"  Số nhãn: {len(label_list)}")
    print(f"  Đang tải dataset...")

    train_dataset = BKEEPhoBERTDataset(TRAIN_JOINT_FILE, tokenizer, label_to_idx)
    dev_dataset   = BKEEPhoBERTDataset(DEV_JOINT_FILE,   tokenizer, label_to_idx)

    print(f"  Train: {len(train_dataset)} câu | Dev: {len(dev_dataset)} câu")

    # Run 04 train nhiều epoch hơn vì label space lớn hơn Run 03
    trainer = Trainer(
        model=model,
        args=get_training_args("./saved_models/run_04", "run_04_phobert_trigger_event_type", epochs=8),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=make_compute_metrics(idx_to_label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n  [Run 04] Kết quả cuối: F1={metrics.get('eval_f1', 0)*100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    os.makedirs("./saved_models", exist_ok=True)

    # Chạy tuần tự Run 03 rồi Run 04
    run_03_phobert_pure()
    run_04_phobert_joint()