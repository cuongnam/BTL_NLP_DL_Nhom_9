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

# Ép buộc chạy đơn card GPU ổn định nhất trên Kaggle
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = "vinai/phobert-base-v2"

class BKEEPhoBERTDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_to_idx, max_len=128):
        self.examples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                item = json.loads(line.strip())
                
                tokens = item["tokens"]
                labels = item["labels"]
                
                # Tokenize có kèm căn chỉnh mảng từ (is_split_into_words=True)
                encoding = tokenizer(
                    tokens,
                    is_split_into_words=True,
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Ánh xạ nhãn BIO theo các sub-tokens của PhoBERT
                word_ids = encoding.word_ids(batch_index=0)
                aligned_labels = []
                for word_idx in word_ids:
                    if word_idx is None:
                        aligned_labels.append(-100) # Bỏ qua các token đặc biệt khi tính Loss
                    else:
                        aligned_labels.append(label_to_idx.get(labels[word_idx], label_to_idx["O"]))
                        
                item_dict = {key: val.squeeze(0) for key, val in encoding.items()}
                item_dict["labels"] = torch.tensor(aligned_labels, dtype=torch.long)
                self.examples.append(item_dict)
                
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
    print("KHỞI CHẠY RUN 03: PHOBERT PURE TRIGGER")
    print("="*50)
    
    label_list = ["O", "B-TRIGGER", "I-TRIGGER"]
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))
    
    train_dataset = BKEEPhoBERTDataset("./data/processed_bio/train_bio.json", tokenizer, label_to_idx)
    dev_dataset = BKEEPhoBERTDataset("./data/processed_bio/dev_bio.json", tokenizer, label_to_idx)
    
    training_args = TrainingArguments(
        output_dir="./saved_models/run_03",
        eval_strategy="epoch",
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3, # Chạy 3 epochs tối ưu tốc độ và tránh Overfitting
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
    print("KHỞI CHẠY RUN 04: PHOBERT JOINT LEARNING")
    print("="*50)
    
    # Tự động thu thập toàn bộ tập nhãn Gộp (Joint) từ tập dữ liệu Giai đoạn 1
    label_set = set(["O"])
    with open("./data/processed_joint/train_joint.json", "r", encoding="utf-8") as f:
        for line in f:
            for l in json.loads(line.strip())["labels"]:
                label_set.add(l)
                
    label_list = sorted(list(label_set))
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
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
    # Đảm bảo đăng nhập W&B trước (Hoặc notebook đã nhận diện qua môi trường)
    run_03_phobert_pure()
    run_04_phobert_joint()