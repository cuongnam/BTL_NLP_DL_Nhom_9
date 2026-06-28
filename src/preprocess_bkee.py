"""
preprocess_bkee.py - Chuyển đổi dữ liệu BKEE sang định dạng BIO

Cấu trúc BKEE (processed/train.json):
  - tokens: list từ (word-level)
  - token_lens: số subword của mỗi token (dùng cho PhoBERT)
  - pieces: subword tokens (▁...) đã tokenize sẵn cho PhoBERT
  - event_mentions[].trigger.start/end: word-level index (không phải char)

Output:
  - processed_bio/   : nhãn B-TRIGGER / I-TRIGGER / O (dùng cho Run 01, 02, 03)
  - processed_joint/ : nhãn B-<EventType> / I-<EventType> / O (dùng cho Run 04)
"""

import json
import os


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def convert_bkee_to_bio(input_path, output_path, joint_learning=False):
    """
    Chuyển BKEE sang BIO.

    Lưu ý quan trọng về BKEE:
      - trigger["start"] và trigger["end"] là WORD-LEVEL index (index vào mảng tokens)
      - Nếu một token bị nhiều sự kiện gán nhãn chồng lên nhau, ưu tiên sự kiện đầu tiên
    """
    print(f"Đang xử lý: {input_path}")

    if not os.path.exists(input_path):
        print(f"  [LỖI] Không tìm thấy file: {input_path}")
        return 0

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    processed_count = 0
    skipped_count = 0

    with open(input_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:

        for line in f_in:
            line = line.strip()
            if not line:
                continue

            doc = json.loads(line)
            tokens = doc.get("tokens", [])
            if not tokens:
                skipped_count += 1
                continue

            # Khởi tạo nhãn mặc định O cho tất cả token
            labels = ["O"] * len(tokens)

            for event in doc.get("event_mentions", []):
                event_type = event.get("event_type", "EVENT")
                trigger = event.get("trigger", {})

                start_idx = trigger.get("start", -1)
                end_idx = trigger.get("end", -1)  # exclusive

                if start_idx < 0 or end_idx < 0:
                    continue

                # Đảm bảo không vượt quá độ dài tokens
                start_idx = min(start_idx, len(tokens) - 1)
                end_idx = min(end_idx, len(tokens))

                suffix = event_type if joint_learning else "TRIGGER"

                # Chỉ gán nếu chưa bị gán (tránh ghi đè)
                if labels[start_idx] == "O":
                    labels[start_idx] = f"B-{suffix}"
                    for i in range(start_idx + 1, end_idx):
                        if labels[i] == "O":
                            labels[i] = f"I-{suffix}"

            output_item = {
                "id": doc.get("sent_id", doc.get("doc_id", "")),
                "tokens": tokens,
                "labels": labels,
                # Giữ lại pieces và token_lens để PhoBERT dùng trực tiếp (tối ưu hơn)
                "pieces": doc.get("pieces", []),
                "token_lens": doc.get("token_lens", []),
            }
            f_out.write(json.dumps(output_item, ensure_ascii=False) + "\n")
            processed_count += 1

    print(f"  -> Xong! Đã xử lý {processed_count} câu, bỏ qua {skipped_count} câu rỗng.")
    print(f"  -> Lưu tại: {output_path}")
    return processed_count


def print_sample(output_path, n=3):
    """In thử n mẫu để kiểm tra kết quả."""
    print(f"\n--- Kiểm tra {n} mẫu đầu từ {output_path} ---")
    with open(output_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            item = json.loads(line.strip())
            tokens = item["tokens"]
            labels = item["labels"]
            # Chỉ in các token có nhãn khác O để dễ đọc
            trigger_tokens = [(t, l) for t, l in zip(tokens, labels) if l != "O"]
            if trigger_tokens:
                print(f"  [{i}] {item['id']}")
                print(f"       Triggers: {trigger_tokens}")
            else:
                print(f"  [{i}] {item['id']} — không có trigger")


if __name__ == "__main__":
    # Đường dẫn dữ liệu gốc BKEE
    DATA_DIR = "./data/processed"
    SPLITS = ["train", "dev", "test"]

    # 1. Tạo dữ liệu BIO thông thường (Run 01, 02, 03)
    print("=" * 60)
    print("TẠO DỮ LIỆU BIO THÔNG THƯỜNG (Run 01 / 02 / 03)")
    print("=" * 60)
    for split in SPLITS:
        convert_bkee_to_bio(
            input_path=f"{DATA_DIR}/{split}.json",
            output_path=f"./data/processed_bio/{split}_bio.json",
            joint_learning=False,
        )

    # 2. Tạo dữ liệu Joint Learning (Run 04)
    print("\n" + "=" * 60)
    print("TẠO DỮ LIỆU JOINT LEARNING (Run 04)")
    print("=" * 60)
    for split in SPLITS:
        convert_bkee_to_bio(
            input_path=f"{DATA_DIR}/{split}.json",
            output_path=f"./data/processed_joint/{split}_joint.json",
            joint_learning=True,
        )

    # Kiểm tra nhanh
    print_sample("./data/processed_bio/train_bio.json")
    print_sample("./data/processed_joint/train_joint.json")

    print("\n[HOÀN TẤT] Tiền xử lý dữ liệu xong!")
    print("  Thư mục output:")
    print("    data/processed_bio/   — dùng cho Run 01, 02, 03")
    print("    data/processed_joint/ — dùng cho Run 04")