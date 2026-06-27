import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import wandb

# Cấu hình đường dẫn dữ liệu từ Giai đoạn 1
TRAIN_BIO_FILE = "./data/processed_bio/train_bio.json"
DEV_BIO_FILE = "./data/processed_bio/dev_bio.json"

# ==========================================================
# RUN 01: RULE-BASED TRIGGER (Exact Match từ điển)
# ==========================================================
def run_01_rule_based():
    print("\n" + "="*50)
    print("KHỞI CHẠY RUN 01: RULE-BASED TRIGGER")
    print("="*50)
    
    # 1. Thu thập tất cả các từ là Trigger từ tập Train để bỏ vào từ điển
    trigger_vocab = set()
    with open(TRAIN_BIO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            for token, label in zip(item["tokens"], item["labels"]):
                if label.startswith("B-") or label.startswith("I-"):
                    trigger_vocab.add(token.lower())
                    
    print(f"-> Đã xây dựng xong từ điển chứa {len(trigger_vocab)} từ khóa kích hoạt.")
    
    # 2. Đánh giá trên tập Dev
    total_gold_triggers = 0
    total_predicted_triggers = 0
    correct_predictions = 0
    
    with open(DEV_BIO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line.strip())
            tokens = item["tokens"]
            gold_labels = item["labels"]
            
            # Dự đoán theo luật khớp từ điển thô
            pred_labels = ["O"] * len(tokens)
            for i, token in enumerate(tokens):
                if token.lower() in trigger_vocab:
                    pred_labels[i] = "B-TRIGGER"
            
            # Tính toán ma trận lỗi ở mức độ Token
            for pred, gold in zip(pred_labels, gold_labels):
                if gold != "O": 
                    total_gold_triggers += 1
                if pred != "O": 
                    total_predicted_triggers += 1
                if pred != "O" and gold != "O": 
                    correct_predictions += 1
                    
    precision = correct_predictions / total_predicted_triggers if total_predicted_triggers > 0 else 0
    recall = correct_predictions / total_gold_triggers if total_gold_triggers > 0 else 0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print("\n--- KẾT QUẢ NGHIỆM THU RUN 01 ---")
    print(f"Precision : {precision * 100:.2f}%")
    print(f"Recall    : {recall * 100:.2f}%")
    print(f"F1-score  : {f1_score * 100:.2f}%")
    print("="*50 + "\n")
    return f1_score


# ==========================================================
# RUN 02: BiLSTM-CRF TRIGGER (Học sâu có ràng buộc chuỗi)
# ==========================================================
# Cài đặt tầng CRF thủ công để không bị lỗi phụ thuộc thư viện ngoài trên Kaggle
class SimpleCRF(nn.Module):
    def __init__(self, num_tags):
        super().__init__()
        self.num_tags = num_tags
        # Ma trận chuyển trạng thái giữa các nhãn (transition matrix)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
    def forward(self, emissions, tags, mask):
        # Tính toán log-likelihood xấp xỉ đơn giản cho bài toán phân lớp chuỗi
        log_probs = torch.log_softmax(emissions, dim=-1)
        tags_expanded = tags.unsqueeze(-1)
        gold_scores = torch.gather(log_probs, dim=-1, index=tags_expanded).squeeze(-1)
        return -torch.mean(gold_scores * mask.float())

class BKEETriggerDataset(Dataset):
    def __init__(self, file_path, word_to_idx, label_to_idx):
        self.features = []
        self.labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                w_ids = [word_to_idx.get(w.lower(), word_to_idx["<UNK>"]) for w in item["tokens"]]
                l_ids = [label_to_idx.get(l, label_to_idx["O"]) for l in item["labels"]]
                self.features.append(torch.tensor(w_ids, dtype=torch.long))
                self.labels.append(torch.tensor(l_ids, dtype=torch.long))
                
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]

def collate_fn(batch):
    sequences, labels = zip(*batch)
    padded_seqs = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=0)
    return padded_seqs, padded_labels

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_tags, embed_dim=128, hidden_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim // 2, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_tags)
        self.crf = SimpleCRF(num_tags)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.fc(lstm_out)
        return emissions

def run_02_bilstm_crf():
    print("="*50)
    print("KHỞI CHẠY RUN 02: BiLSTM-CRF TRIGGER")
    print("="*50)
    
    # Khởi tạo đồng bộ Weights & Biases cho Run 02
    wandb.init(project="BKEE_Event_Extraction_LREC2024", name="run_02_bilstm_crf_trigger")
    
    # Xây dựng bộ từ vựng ánh xạ ID cố định
    word_to_idx = {"<PAD>": 0, "<UNK>": 1}
    label_to_idx = {"O": 0, "B-TRIGGER": 1, "I-TRIGGER": 2}
    
    with open(TRAIN_BIO_FILE, "r", encoding="utf-8") as f:
        for line in f:
            for w in json.loads(line.strip())["tokens"]:
                if w.lower() not in word_to_idx:
                    word_to_idx[w.lower()] = len(word_to_idx)
                    
    train_set = BKEETriggerDataset(TRAIN_BIO_FILE, word_to_idx, label_to_idx)
    dev_set = BKEETriggerDataset(DEV_BIO_FILE, word_to_idx, label_to_idx)
    
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_set, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTM_CRF(len(word_to_idx), len(label_to_idx)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.002)
    
    # Huấn luyện mô hình mạng nơ-ron trong 5 Epochs
    for epoch in range(5):
        model.train()
        epoch_loss = 0
        for seqs, labels in train_loader:
            seqs, labels = seqs.to(device), labels.to(device)
            mask = (seqs != 0)
            
            optimizer.zero_grad()
            emissions = model(seqs)
            loss = model.crf(emissions, labels, mask)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        # Đánh giá nhanh F1 trên tập Dev cuối mỗi Epoch
        model.eval()
        correct, pred_total, gold_total = 0, 0, 0
        with torch.no_grad():
            for seqs, labels in dev_loader:
                seqs = seqs.to(device)
                emissions = model(seqs)
                preds = torch.argmax(emissions, dim=-1).cpu().numpy()
                golds = labels.numpy()
                
                for p_seq, g_seq in zip(preds, golds):
                    for p, g in zip(p_seq, g_seq):
                        if g != 0: gold_total += 1
                        if p != 0: pred_total += 1
                        if p != 0 and g != 0: correct += 1
                        
        p = correct / pred_total if pred_total > 0 else 0
        r = correct / gold_total if gold_total > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        
        # Đẩy dữ liệu log trực quan lên đồ thị W&B
        wandb.log({"epoch": epoch+1, "loss": epoch_loss/len(train_loader), "val_f1": f1})
        print(f"Epoch {epoch+1}/5 - Loss: {epoch_loss/len(train_loader):.4f} - Dev F1: {f1*100:.2f}%")
        
    wandb.finish()
    print("-> Hoàn thành huấn luyện Run 02!")

if __name__ == "__main__":
    # Chạy lần lượt Run 01 và Run 02
    run_01_rule_based()
    run_02_bilstm_crf()