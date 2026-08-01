"""
build_database.py
------------------
Đọc file data/memories.json, tạo vector embedding cho từng đoạn ký ức
bằng model tiếng Việt, rồi lưu vào ChromaDB (vector database chạy local,
không cần server riêng).

Chạy 1 lần mỗi khi dataset thay đổi (thêm/sửa ký ức):
    python build_database.py
"""

import json
import chromadb
from sentence_transformers import SentenceTransformer

DATA_PATH = "data/memories.json"
DB_PATH = "./memory_db"
COLLECTION_NAME = "living_memory"

# Model embedding tiếng Việt - KHÔNG dùng model tiếng Anh mặc định
# vì độ chính xác kém hẳn với câu hỏi tiếng Việt.
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"


def main():
    print(f"Đang tải model embedding: {EMBEDDING_MODEL} ...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"Đang đọc dataset: {DATA_PATH} ...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        dataset = json.load(f)
    print(f"  -> {len(dataset)} đoạn ký ức.")

    print(f"Khởi tạo ChromaDB tại: {DB_PATH} ...")
    client = chromadb.PersistentClient(path=DB_PATH)

    # Xóa collection cũ nếu có, để build lại từ đầu cho sạch
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    ids, embeddings, documents, metadatas = [], [], [], []

    for item in dataset:
        text = item["noi_dung"]
        emb = model.encode(text).tolist()

        ids.append(item["id"])
        embeddings.append(emb)
        documents.append(text)
        metadatas.append({
            "nhan_chung": item.get("nhan_chung", ""),
            "chu_de": item.get("chu_de", ""),
            "audio_path": item.get("audio_path", ""),
            "audio_start": item.get("audio_start", 0),
            "audio_end": item.get("audio_end", 0),
            "nguon": item.get("nguon", ""),
            "xac_thuc": item.get("xac_thuc", False),
        })

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=documents,
        metadatas=metadatas,
    )

    print(f"Đã nạp {len(ids)} đoạn ký ức vào vector database.")
    print("Xong! Giờ có thể chạy: streamlit run app.py")


if __name__ == "__main__":
    main()
