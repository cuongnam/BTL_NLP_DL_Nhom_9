
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
import wandb

# Khóa cứng tiến trình vào card GPU số 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
MODEL_NAME = "vinai/phobert-base-v2"
WANDB_PROJECT = "BKEE_Event_Extraction_LREC2024"

# ==========================================
# 1. TIỀN XỬ LÝ DỮ LIỆU THEO CẤU TRÚC MRC
# ==========================================
class BKEEArgumentMRCDataset(Dataset):
    def __init__(self, file_path, tokenizer, label_to_idx, max_len=256):
        self.examples = []
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip(): continue
                doc = json.loads(line.strip())
                tokens = doc.get("tokens", [])
                
                # Mỗi sự kiện trong câu sẽ được tạo thành một mẫu huấn luyện riêng biệt
                for event in doc.get("event_mentions", []):
                    event_type = event["event_type"]
                    trigger = event["trigger"]
                    trigger_text = " ".join(tokens[trigger["start"]:trigger["end"]])
                    
                    # 1. Tạo chuỗi mồi (Prompt) cung cấp ngữ cảnh cho mô hình
                    prompt_text = f"Sự kiện: {event_type} | Từ khóa: {trigger_text}"
                    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
                    
                    # 2. Xử lý câu văn gốc và gán nhãn BIO cho các Argument (Đối số)
                    sentence_input_ids = []
                    sentence_label_ids = []
                    
                    # Khởi tạo nhãn 'O' cho toàn bộ câu
                    arg_labels = ["O"] * len(tokens)
                    for arg in event.get("arguments", []):
                        role = arg["role"]
                        start, end = arg["start"], arg["end"]
                        if start < len(arg_labels):
                            arg_labels[start] = f"B-{role}"
                            for i in range(start + 1, end):
                                if i < len(arg_labels):
                                    arg_labels[i] = f"I-{role}"
                                    
                    # Ánh xạ sub-words thủ công (Tránh lỗi Fast Tokenizer của PhoBERT)
                    for word, label in zip(tokens, arg_labels):
                        word_tokens = tokenizer.encode(word, add_special_tokens=False)
                        if not word_tokens: continue
                        sentence_input_ids.extend(word_tokens)
                        
                        # Gán nhãn gốc cho sub-token đầu tiên, các mảnh sau gán -100
                        sentence_label_ids.append(label_to_idx.get(label, label_to_idx["O"]))
                        if len(word_tokens) > 1:
                            sentence_label_ids.extend([-100] * (len(word_tokens) - 1))
                            
                    # 3. Ghép nối chuẩn xác: [CLS] Prompt [SEP] Sentence [SEP]
                    input_ids = [tokenizer.cls_token_id] + prompt_tokens + [tokenizer.sep_token_id] + sentence_input_ids + [tokenizer.sep_token_id]
                    
                    # Label cho phần prompt và các token đặc biệt là -100 (Không tính Loss)
                    label_ids = [-100] * (len(prompt_tokens) + 2) + sentence_label_ids + [-100]
                    
                    # Cắt gọt và Padding theo max_len
                    input_ids = input_ids[:max_len]
                    label_ids = label_ids[:max_len]
                    
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

# ==========================================
# 2. HÀM QUÉT NHÃN VÀ METRICS
# ==========================================
def extract_argument_labels(file_path):
    """Quét dữ liệu thô để thu thập toàn bộ nhãn Argument có thể có."""
    labels = set(["O"])
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            doc = json.loads(line.strip())
            for event in doc.get("event_mentions", []):
                for arg in event.get("arguments", []):
                    role = arg["role"]
                    labels.add(f"B-{role}")
                    labels.add(f"I-{role}")
    return sorted(list(labels))

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

# ==========================================
# 3. CHẠY RUN 05 VÀ TẠO MA TRẬN NHẦM LẪN
# ==========================================
def run_05_argument_extraction():
    print("\n" + "="*50)
    print("KHỞI CHẠY RUN 05: ARGUMENT EXTRACTION (MRC FORMAT)")
    print("="*50)
    
    TRAIN_RAW = "./data/processed/train.json"
    DEV_RAW = "./data/processed/dev.json"
    
    label_list = extract_argument_labels(TRAIN_RAW)
    label_to_idx = {l: i for i, l in enumerate(label_list)}
    idx_to_label = {i: l for i, l in enumerate(label_list)}
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(label_list))
    
    train_dataset = BKEEArgumentMRCDataset(TRAIN_RAW, tokenizer, label_to_idx)
    dev_dataset = BKEEArgumentMRCDataset(DEV_RAW, tokenizer, label_to_idx)
    
    training_args = TrainingArguments(
        output_dir="./saved_models/run_05",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1, # Lưu best model duy nhất
        learning_rate=3e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=50,
        report_to="wandb",
        run_name="run_05_best_model_argument_extraction" # Tên run yêu cầu
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        compute_metrics=get_compute_metrics(idx_to_label),
    )
    
    # 3.1. Bắt đầu huấn luyện
    trainer.train()
    
    # 3.2. Dự đoán trên tập Dev để lấy dữ liệu vẽ Ma trận nhầm lẫn (Yêu cầu nâng cao 2)
    print("\n[Tiến hành vẽ Ma trận nhầm lẫn cho tập Dev...]")
    predictions_output = trainer.predict(dev_dataset)
    predictions = np.argmax(predictions_output.predictions, axis=-1)
    labels = predictions_output.label_ids
    
    # Làm phẳng mảng 2D thành 1D và loại bỏ các token nhãn -100
    y_true_flat = []
    y_pred_flat = []
    for pred_seq, label_seq in zip(predictions, labels):
        for p_val, l_val in zip(pred_seq, label_seq):
            if l_val != -100:
                y_true_flat.append(idx_to_label[l_val])
                y_pred_flat.append(idx_to_label[p_val])
                
    # Lọc bỏ nhãn 'O' (Outside) để biểu đồ không bị nhiễu và chỉ tập trung vào nhãn 'B-'
    filtered_labels = [l for l in label_list if l != "O" and l.startswith("B-")]
    
    y_true_clean = []
    y_pred_clean = []
    for t, p in zip(y_true_flat, y_pred_flat):
        if t != "O" or p != "O": # Bỏ qua các cặp (O, O)
            t_clean = t.replace("I-", "B-")
            p_clean = p.replace("I-", "B-")
            y_true_clean.append(t_clean if t_clean in filtered_labels else "O")
            y_pred_clean.append(p_clean if p_clean in filtered_labels else "O")
            
    final_labels = filtered_labels + ["O"]
    cm = confusion_matrix(y_true_clean, y_pred_clean, labels=final_labels)
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=final_labels, yticklabels=final_labels)
    plt.ylabel('Thực tế (True Argument)')
    plt.xlabel('Dự đoán (Predicted Argument)')
    plt.title('Ma trận Nhầm lẫn (Confusion Matrix) - Trích xuất Đối số')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    # Lưu ảnh ra file và đẩy lên W&B
    plt.savefig("./confusion_matrix_arguments.png", dpi=300)
    print("-> Đã lưu ảnh Ma trận nhầm lẫn tại: ./confusion_matrix_arguments.png")
    
    wandb.log({"Confusion_Matrix": wandb.Image("./confusion_matrix_arguments.png")})
    wandb.finish()

if __name__ == "__main__":
    run_05_argument_extraction()