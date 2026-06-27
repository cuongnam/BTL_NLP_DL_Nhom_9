import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = "vinai/phobert-base-v2"

# Đã thay đổi: Tự động căn chỉnh nhãn thủ công không dùng word_ids()
class BKEEPhoBERTDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_to_idx, max_len=128):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line.strip())
                
                tokens = item["tokens"]
                labels = item["labels"]
                
                # Khởi tạo chuỗi ID với token [CLS] ở đầu
                input_ids = [tokenizer.cls_token_id]
                label_ids = [-100] # -100 để hàm Loss của PyTorch bỏ qua
                
                # Căn chỉnh nhãn thủ công cho từng từ
                for word, label in zip(tokens, labels):
                    # Tách từ thành các sub-tokens
                    word_tokens = tokenizer.encode(word, add_special_tokens=False)
                    if not word_tokens:
                        continue
                        
                    input_ids.extend(word_tokens)
                    
                    # Gán nhãn gốc cho sub-token đầu tiên của từ
                    label_ids.append(label_to_idx.get(label, label_to_idx["O"]))
                    
                    # Gán nhãn -100 cho các sub-tokens còn lại (nếu từ bị tách làm nhiều mảnh)
                    if len(word_tokens) > 1:
                        label_ids.extend([-100] * (len(word_tokens) - 1))
                        
                # Thêm token [SEP] ở cuối
                input_ids.append(tokenizer.sep_token_id)
                label_ids.append(-100)
                
                # Cắt gọn (Truncation) nếu vượt quá max_len
                input_ids = input_ids[:max_len]
                label_ids = label_ids[:max_len]
                
                # Padding cho đủ max_len
                attention_mask = [1] * len(input_ids)
                pad_len = max_len - len(input_ids)
                
                input_ids.extend([tokenizer.pad_token_id] * pad_len)
                label_ids.extend([-100] * pad_len)
                attention_mask.extend([0] * pad_len)
                
                self.examples.append({
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    "labels": torch.tensor(label_ids, dtype=torch.long)
                })
                
    def __len__(self): return len(self.examples)
    def __getitem__(self, idx): return self.examples[idx]

def get_compute_metrics(idx_to_label):
    seqeval_metric = evaluate.load("seqeval")
    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=-1)
        
        true_predictions = [
            [idx_to_label[p_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [idx_to_label[l_val] for (p_val, l_val) in zip(prediction, label) if l_val != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = seqeval_metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
        }
    return compute_metrics

def run_03_phobert_pure():
    print("\n" + "="*50)
    print("KHỞI CHẠY RUN 03: PHOBERT PURE TRIGGER (TOKEN CLASSIFICATION)")
    print("="*50)
    
    label_list = ["O", "B-TRIGGER", "I-TRIGGER"]
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))
    
    train_dataset = BKEEPhoBERTDataset("./data/processed_bio/train_bio.json", tokenizer, label_to_idx)
    dev_dataset = BKEEPhoBERTDataset("./data/processed_bio/dev_bio.json", tokenizer, label_to_idx)
    
    training_args = TrainingArguments(
        output_dir="./saved_models/run_03",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
        run_name="run_03_phobert_trigger"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=get_compute_metrics(idx_to_label),
    )
    
    trainer.train()
    wandb.finish()

def run_04_phobert_joint():
    print("\n" + "="*50)
    print("KHỞI CHẠY RUN 04: PHOBERT JOINT LEARNING (TRIGGER + EVENT TYPE)")
    print("="*50)
    
    label_set = set(["O"])
    with open("./data/processed_joint/train_joint.json", "r", encoding="utf-8") as f:
        for line in f:
            for l in json.loads(line.strip())["labels"]:
                label_set.add(l)
                
    label_list = sorted(list(label_set))
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))
    
    train_dataset = BKEEPhoBERTDataset("./data/processed_joint/train_joint.json", tokenizer, label_to_idx)
    dev_dataset = BKEEPhoBERTDataset("./data/processed_joint/dev_joint.json", tokenizer, label_to_idx)
    
    training_args = TrainingArguments(
        output_dir="./saved_models/run_04",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
        run_name="run_04_phobert_trigger_event_type"
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=get_compute_metrics(idx_to_label),
    )
    
    trainer.train()
    wandb.finish()

if __name__ == "__main__":
    run_03_phobert_pure()
    run_04_phobert_joint()