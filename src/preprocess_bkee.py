import json
import os

def convert_bkee_to_bio(input_path, output_path, joint_learning=False):
    """
    Chuyển đổi dữ liệu gốc BKEE sang dạng chuỗi token kèm nhãn BIO phù hợp cho các mô hình.
    - joint_learning=False: Nhãn dạng B-TRIGGER, I-TRIGGER, O
    - joint_learning=True: Nhãn dạng B-[Event_Type], I-[Event_Type], O
    """
    print(f"Đang xử lý file dữ liệu: {input_path}")
    if not os.path.exists(input_path):
        print(f"Lỗi dữ liệu: Không tìm thấy file tại {input_path}")
        return

    processed_count = 0
    with open(input_path, "r", encoding="utf-8") as f_in, open(output_path, "w", encoding="utf-8") as f_out:
        for line in f_in:
            if not line.strip(): 
                continue
            
            doc = json.loads(line.strip())
            tokens = doc.get("tokens", [])
            
            # Khởi tạo chuỗi nhãn nền mặc định là 'O' (Outside) cho mọi token
            labels = ["O"] * len(tokens)
            
            # Duyệt qua danh sách các sự kiện được gán nhãn trong câu văn này
            for event in doc.get("event_mentions", []):
                event_type = event["event_type"]
                trigger = event["trigger"]
                
                # Trích xuất chỉ số từ bắt đầu và kết thúc của Trigger
                start_idx = trigger["start"]
                end_idx = trigger["end"] # Vị trí kết thúc (không bao gồm trong nhãn)
                
                # Định nghĩa hậu tố nhãn dựa theo yêu cầu từng Run
                suffix = event_type if joint_learning else "TRIGGER"
                
                # Tiến hành gán nhãn chuỗi dạng BIO dựa trên index từ
                if start_idx < len(labels):
                    labels[start_idx] = f"B-{suffix}"
                    for i in range(start_idx + 1, min(end_idx, len(labels))):
                        labels[i] = f"I-{suffix}"
            
            # Đóng gói dữ liệu sạch thành một dòng JSON mới
            output_item = {
                "id": doc.get("id", ""),
                "tokens": tokens,
                "labels": labels
            }
            f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            processed_count += 1
            
    print(f"-> Đã xử lý xong! File đích: {output_path} (Tổng số câu: {processed_count})")

if __name__ == "__main__":
    # Đường dẫn file gốc của bạn trên Kaggle (Hãy chắc chắn đường dẫn này khớp với vị trí file của bạn)
    ORIGINAL_TRAIN = "./data/processed/train.json"
    ORIGINAL_DEV = "./data/processed/dev.json"
    
    # 1. Tạo dữ liệu nhãn Trigger độc lập (Dùng cho Run 01, 02, 03)
    convert_bkee_to_bio(ORIGINAL_TRAIN, "./data/processed_bio/train_bio.json", joint_learning=False)
    convert_bkee_to_bio(ORIGINAL_DEV, "./data/processed_bio/dev_bio.json", joint_learning=False)
    
    # 2. Tạo dữ liệu nhãn Gộp Joint Learning (Dùng cho Run 04)
    convert_bkee_to_bio(ORIGINAL_TRAIN, "./data/processed_joint/train_joint.json", joint_learning=True)
    convert_bkee_to_bio(ORIGINAL_DEV, "./data/processed_joint/dev_joint.json", joint_learning=True)