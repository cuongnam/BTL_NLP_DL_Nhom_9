"""
train_phobert_model.py - Run 03 (PhoBERT) và Run 04 (PhoBERT Joint)

FIX so với lần trước (giải quyết Recall ~ 0):
  1. Thêm WeightedTrainer dùng class-weighted cross-entropy loss
     để chống class imbalance (O chiếm ~88% tokens)
  2. Tăng learning_rate 3e-5 -> 5e-5
  3. Giảm warmup_ratio 0.1 -> 0.06
  4. Thêm hàm compute_class_weights() tự động từ train data
  5. Giữ nguyên label list cố định, pieces/token_lens optimization, early stopping

Yêu cầu:
  pip install transformers torch evaluate seqeval wandb
"""

import json
import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
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
MAX_LEN       = 256

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


# ─── Tính class weights từ train data ────────────────────────────────────────

def compute_class_weights(train_file, label_to_idx):
    """
    Tính inverse-frequency weight cho mỗi class.
    Token 'O' chiếm ~88% → weight thấp.
    Token 'B-TRIGGER', 'I-TRIGGER' → weight cao hơn nhiều.
    """
    counter = Counter()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            for lbl in json.loads(line.strip())["labels"]:
                counter[lbl] += 1

    total = sum(counter.values())
    num_labels = len(label_to_idx)
    weights = torch.ones(num_labels)

    for label, idx in label_to_idx.items():
        count = counter.get(label, 1)
        # Inverse frequency, chuẩn hóa theo tổng số class
        weights[idx] = total / (num_labels * count)

    # Chuẩn hóa để weight trung bình = 1.0
    weights = weights / weights.mean()

    print("  Class weights:")
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"    {label:25s} count={counter.get(label,0):6d}  weight={weights[idx]:.3f}")

    return weights


# ─── Custom Trainer với weighted loss ────────────────────────────────────────

class WeightedTrainer(Trainer):
    """
    Override compute_loss để dùng class-weighted cross-entropy.
    Giải quyết class imbalance: token 'O' chiếm ~88% → mô hình thường
    collapse về predict toàn 'O', dẫn đến Recall gần 0.
    """

    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights  # tensor shape [num_labels]

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits  = outputs.logits  # [batch, seq_len, num_labels]

        device = logits.device
        if self.class_weights is not None:
            weight = self.class_weights.to(device)
        else:
            weight = None

        loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        # Reshape: [batch*seq_len, num_labels] vs [batch*seq_len]
        loss = loss_fn(
            logits.view(-1, logits.shape[-1]),
            labels.view(-1),
        )

        return (loss, outputs) if return_outputs else loss


# ─── Label list cố định cho Run 04 ───────────────────────────────────────────

def build_joint_label_list(train_joint_path):
    label_set = set()
    with open(train_joint_path, "r", encoding="utf-8") as f:
        for line in f:
            for lbl in json.loads(line.strip())["labels"]:
                label_set.add(lbl)

    label_set.discard("O")
    event_types = sorted(set(l[2:] for l in label_set if l.startswith("B-")))
    ordered = ["O"]
    for et in event_types:
        if f"B-{et}" in label_set:
            ordered.append(f"B-{et}")
        if f"I-{et}" in label_set:
            ordered.append(f"I-{et}")

    print(f"  Joint label list: {len(ordered)} nhãn")
    return ordered


# ─── Dataset ─────────────────────────────────────────────────────────────────

class BKEEPhoBERTDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_to_idx, max_len=MAX_LEN):
        self.examples = []

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

                if pieces and token_lens and len(token_lens) == len(tokens):
                    example = self._use_pieces(pieces, token_lens, labels, label_to_idx, tokenizer, max_len)
                else:
                    example = self._tokenize_fallback(tokens, labels, label_to_idx, tokenizer, max_len)

                if example is not None:
                    self.examples.append(example)

    def _use_pieces(self, pieces, token_lens, labels, label_to_idx, tokenizer, max_len):
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        input_ids = [cls_id]
        label_ids = [-100]

        piece_idx = 0
        for n_pieces, label in zip(token_lens, labels):
            token_pieces = pieces[piece_idx: piece_idx + n_pieces]
            piece_idx   += n_pieces
            piece_ids    = tokenizer.convert_tokens_to_ids(token_pieces)
            if not piece_ids:
                continue
            input_ids.append(piece_ids[0])
            label_ids.append(label_to_idx.get(label, label_to_idx["O"]))
            for pid in piece_ids[1:]:
                input_ids.append(pid)
                label_ids.append(-100)

        input_ids.append(sep_id)
        label_ids.append(-100)
        return self._pad(input_ids, label_ids, max_len, pad_id)

    def _tokenize_fallback(self, tokens, labels, label_to_idx, tokenizer, max_len):
        cls_id = tokenizer.cls_token_id
        sep_id = tokenizer.sep_token_id
        pad_id = tokenizer.pad_token_id

        input_ids = [cls_id]
        label_ids = [-100]

        for word, label in zip(tokens, labels):
            word_ids = tokenizer.encode(word, add_special_tokens=False)
            if not word_ids:
                continue
            input_ids.append(word_ids[0])
            label_ids.append(label_to_idx.get(label, label_to_idx["O"]))
            for wid in word_ids[1:]:
                input_ids.append(wid)
                label_ids.append(-100)

        input_ids.append(sep_id)
        label_ids.append(-100)
        return self._pad(input_ids, label_ids, max_len, pad_id)

    def _pad(self, input_ids, label_ids, max_len, pad_id):
        input_ids = input_ids[:max_len]
        label_ids = label_ids[:max_len]
        attn_mask = [1] * len(input_ids)
        pad_len   = max_len - len(input_ids)
        input_ids += [pad_id] * pad_len
        label_ids += [-100]   * pad_len
        attn_mask += [0]      * pad_len
        return {
            "input_ids":      torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels":         torch.tensor(label_ids, dtype=torch.long),
        }

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        return self.examples[idx]


# ─── Compute metrics ─────────────────────────────────────────────────────────

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
        learning_rate=5e-5,          # tăng từ 3e-5 lên 5e-5
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,           # giảm từ 0.1 xuống 0.06
        logging_steps=50,
        report_to="wandb",
        run_name=run_name,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 03 — PhoBERT + Weighted Loss
# ═══════════════════════════════════════════════════════════════════════════════

def run_03_phobert_pure():
    print("\n" + "=" * 60)
    print("KHỞI CHẠY RUN 03 (v2): PhoBERT + Weighted Cross-Entropy")
    print("=" * 60)

    set_seed()

    label_list   = ["O", "B-TRIGGER", "I-TRIGGER"]
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}

    os.makedirs("./saved_models/run_03", exist_ok=True)
    with open("./saved_models/run_03/label_list.json", "w") as f:
        json.dump(label_list, f)

    # Tính class weights
    class_weights = compute_class_weights(TRAIN_BIO_FILE, label_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=idx_to_label,
        label2id=label_to_idx,
    )

    print(f"\n  Đang tải dataset...")
    train_dataset = BKEEPhoBERTDataset(TRAIN_BIO_FILE, tokenizer, label_to_idx)
    dev_dataset   = BKEEPhoBERTDataset(DEV_BIO_FILE,   tokenizer, label_to_idx)
    print(f"  Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=get_training_args("./saved_models/run_03", "run_03_phobert_weighted", epochs=5),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=make_compute_metrics(idx_to_label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n  [Run 03 v2] F1={metrics.get('eval_f1',0)*100:.2f}%  "
          f"P={metrics.get('eval_precision',0)*100:.2f}%  "
          f"R={metrics.get('eval_recall',0)*100:.2f}%")
    wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 04 — PhoBERT Joint + Weighted Loss
# ═══════════════════════════════════════════════════════════════════════════════

def run_04_phobert_joint():
    print("\n" + "=" * 60)
    print("KHỞI CHẠY RUN 04 (v2): PhoBERT Joint + Weighted Cross-Entropy")
    print("=" * 60)

    set_seed()

    label_list   = build_joint_label_list(TRAIN_JOINT_FILE)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}

    os.makedirs("./saved_models/run_04", exist_ok=True)
    with open("./saved_models/run_04/label_list.json", "w") as f:
        json.dump(label_list, f)

    # Tính class weights — quan trọng hơn ở Run 04 vì 33+ class, nhiều class rất hiếm
    class_weights = compute_class_weights(TRAIN_JOINT_FILE, label_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(label_list),
        id2label=idx_to_label,
        label2id=label_to_idx,
    )

    print(f"\n  Đang tải dataset...")
    train_dataset = BKEEPhoBERTDataset(TRAIN_JOINT_FILE, tokenizer, label_to_idx)
    dev_dataset   = BKEEPhoBERTDataset(DEV_JOINT_FILE,   tokenizer, label_to_idx)
    print(f"  Train: {len(train_dataset)} | Dev: {len(dev_dataset)}")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=get_training_args("./saved_models/run_04", "run_04_phobert_joint_weighted", epochs=8),
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=make_compute_metrics(idx_to_label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print(f"\n  [Run 04 v2] F1={metrics.get('eval_f1',0)*100:.2f}%  "
          f"P={metrics.get('eval_precision',0)*100:.2f}%  "
          f"R={metrics.get('eval_recall',0)*100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    os.makedirs("./saved_models", exist_ok=True)
    run_03_phobert_pure()
    run_04_phobert_joint()