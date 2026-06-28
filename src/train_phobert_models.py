"""
train_phobert_model.py - Run 03 (PhoBERT) và Run 04 (PhoBERT Joint)  [v3]

Thay đổi so với v2 (fix F1=0):
  - Bỏ WeightedTrainer.compute_loss dùng inputs.pop() — không tương thích transformers >= 4.40
  - Dùng cách an toàn hơn: override compute_loss với model(**inputs) rồi tính loss riêng
  - Class weight được CAP ở mức tối đa 5.0 để tránh over-correction
  - Tăng learning_rate lên 5e-5, giảm warmup_ratio xuống 0.06

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
MAX_WEIGHT    = 5.0   # cap để tránh class weight quá lớn gây instability

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


# ─── Class weights ────────────────────────────────────────────────────────────

def compute_class_weights(train_file, label_to_idx, max_weight=MAX_WEIGHT):
    counter = Counter()
    with open(train_file, "r", encoding="utf-8") as f:
        for line in f:
            for lbl in json.loads(line.strip())["labels"]:
                counter[lbl] += 1

    total      = sum(counter.values())
    num_labels = len(label_to_idx)
    weights    = torch.ones(num_labels)

    for label, idx in label_to_idx.items():
        count = counter.get(label, 1)
        w     = total / (num_labels * count)
        weights[idx] = min(w, max_weight)   # CAP để tránh over-correction

    # Normalize về mean=1
    weights = weights / weights.mean()

    print("\n  Class weights (sau khi cap và normalize):")
    for label, idx in sorted(label_to_idx.items(), key=lambda x: x[1]):
        print(f"    {label:30s}  count={counter.get(label,0):6d}  weight={weights[idx]:.3f}")

    return weights


# ─── WeightedTrainer — cách an toàn với transformers >= 4.40 ─────────────────

class WeightedTrainer(Trainer):
    """
    Override compute_loss theo cách tương thích với transformers >= 4.40:
    KHÔNG dùng inputs.pop() mà clone inputs, để labels nguyên trong inputs
    rồi tính loss thủ công từ logits.
    """
    def __init__(self, class_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Truyền toàn bộ inputs vào model (bao gồm labels để model không báo lỗi)
        outputs = model(**inputs)
        logits  = outputs.logits          # [batch, seq_len, num_labels]
        labels  = inputs["labels"]        # đọc, KHÔNG pop

        device = logits.device
        weight = self.class_weights.to(device) if self.class_weights is not None else None

        loss_fct = nn.CrossEntropyLoss(weight=weight, ignore_index=-100)
        loss = loss_fct(
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
        if f"B-{et}" in label_set: ordered.append(f"B-{et}")
        if f"I-{et}" in label_set: ordered.append(f"I-{et}")

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
                    ex = self._use_pieces(pieces, token_lens, labels, label_to_idx, tokenizer, max_len)
                else:
                    ex = self._tokenize_fallback(tokens, labels, label_to_idx, tokenizer, max_len)

                if ex is not None:
                    self.examples.append(ex)

    def _use_pieces(self, pieces, token_lens, labels, label_to_idx, tokenizer, max_len):
        input_ids = [tokenizer.cls_token_id]
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
        input_ids.append(tokenizer.sep_token_id)
        label_ids.append(-100)
        return self._pad(input_ids, label_ids, max_len, tokenizer.pad_token_id)

    def _tokenize_fallback(self, tokens, labels, label_to_idx, tokenizer, max_len):
        input_ids = [tokenizer.cls_token_id]
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
        input_ids.append(tokenizer.sep_token_id)
        label_ids.append(-100)
        return self._pad(input_ids, label_ids, max_len, tokenizer.pad_token_id)

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

    def __len__(self):  return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]


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
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        report_to="wandb",
        run_name=run_name,
        seed=SEED,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=2,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 03
# ═══════════════════════════════════════════════════════════════════════════════

def run_03_phobert_pure():
    print("\n" + "=" * 60)
    print("RUN 03 (v3): PhoBERT + Weighted Loss (safe version)")
    print("=" * 60)
    set_seed()

    label_list   = ["O", "B-TRIGGER", "I-TRIGGER"]
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}

    os.makedirs("./saved_models/run_03", exist_ok=True)
    with open("./saved_models/run_03/label_list.json", "w") as f:
        json.dump(label_list, f)

    class_weights = compute_class_weights(TRAIN_BIO_FILE, label_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list),
        id2label=idx_to_label, label2id=label_to_idx,
    )

    print("\n  Đang tải dataset...")
    train_ds = BKEEPhoBERTDataset(TRAIN_BIO_FILE, tokenizer, label_to_idx)
    dev_ds   = BKEEPhoBERTDataset(DEV_BIO_FILE,   tokenizer, label_to_idx)
    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=get_training_args("./saved_models/run_03", "run_03_phobert_weighted_v3", epochs=5),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=make_compute_metrics(idx_to_label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    m = trainer.evaluate()
    print(f"\n  [Run 03 v3] F1={m.get('eval_f1',0)*100:.2f}%  "
          f"P={m.get('eval_precision',0)*100:.2f}%  R={m.get('eval_recall',0)*100:.2f}%")
    wandb.finish()


# ═══════════════════════════════════════════════════════════════════════════════
# RUN 04
# ═══════════════════════════════════════════════════════════════════════════════

def run_04_phobert_joint():
    print("\n" + "=" * 60)
    print("RUN 04 (v3): PhoBERT Joint + Weighted Loss (safe version)")
    print("=" * 60)
    set_seed()

    label_list   = build_joint_label_list(TRAIN_JOINT_FILE)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}

    os.makedirs("./saved_models/run_04", exist_ok=True)
    with open("./saved_models/run_04/label_list.json", "w") as f:
        json.dump(label_list, f)

    class_weights = compute_class_weights(TRAIN_JOINT_FILE, label_to_idx)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model     = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME, num_labels=len(label_list),
        id2label=idx_to_label, label2id=label_to_idx,
    )

    print("\n  Đang tải dataset...")
    train_ds = BKEEPhoBERTDataset(TRAIN_JOINT_FILE, tokenizer, label_to_idx)
    dev_ds   = BKEEPhoBERTDataset(DEV_JOINT_FILE,   tokenizer, label_to_idx)
    print(f"  Train: {len(train_ds)} | Dev: {len(dev_ds)}")

    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=get_training_args("./saved_models/run_04", "run_04_phobert_joint_weighted_v3", epochs=8),
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        compute_metrics=make_compute_metrics(idx_to_label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    m = trainer.evaluate()
    print(f"\n  [Run 04 v3] F1={m.get('eval_f1',0)*100:.2f}%  "
          f"P={m.get('eval_precision',0)*100:.2f}%  R={m.get('eval_recall',0)*100:.2f}%")
    wandb.finish()


if __name__ == "__main__":
    os.makedirs("./saved_models", exist_ok=True)
    run_03_phobert_pure()
    run_04_phobert_joint()