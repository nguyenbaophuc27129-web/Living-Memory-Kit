# Living Memory Kit — AI Demo (Streamlit + RAG)

Demo kỹ thuật: chat AI trả lời dựa trên ký ức thật đã số hóa, theo đúng mô hình
**RAG (Retrieval-Augmented Generation)** — không "train" lại AI, chỉ tra cứu +
đưa ngữ cảnh đúng cho AI mỗi lần trả lời.

## 1. Cài đặt (chỉ làm 1 lần)

```bash
# Tạo môi trường ảo (khuyến nghị)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Cài thư viện
pip install -r requirements.txt
```

## 2. Lấy API key của Claude

1. Vào **console.anthropic.com** → đăng ký tài khoản (có gói dùng thử miễn phí).
2. Vào mục **API Keys** → tạo key mới.
3. Cấu hình key vào biến môi trường:

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxx"

# Windows (PowerShell)
setx ANTHROPIC_API_KEY "sk-ant-xxxxxxxx"
```

> Không commit API key lên GitHub công khai — key bị lộ có thể bị người khác dùng và tính phí vào tài khoản của bạn.

## 3. Chuẩn bị dữ liệu (dataset)

Sửa file `data/memories.json` — mỗi đoạn ký ức là một object theo mẫu có sẵn.
File hiện tại đang chứa **dữ liệu mẫu (demo)**, có ghi rõ `"xac_thuc": false` —
hãy thay bằng dữ liệu thật đã xác thực khi có, và đổi thành `"xac_thuc": true`.

Nếu có file ghi âm thô cần lọc nhiễu trước:
```bash
python clean_audio.py raw_audio/phong_van.wav audio/demo_001_clean.wav
```

## 4. Build vector database

Chạy lệnh này mỗi khi thêm/sửa dữ liệu trong `memories.json`:
```bash
python build_database.py
```

## 5. Chạy app

```bash
streamlit run app.py
```

Trình duyệt sẽ tự mở tại `http://localhost:8501`.

## 6. Cấu trúc thư mục

```
living-memory-ai/
├── app.py                 # Giao diện chat Streamlit
├── build_database.py      # Script tạo vector database từ dataset
├── clean_audio.py         # Script lọc nhiễu audio (FFT)
├── requirements.txt
├── data/
│   └── memories.json       # Dataset ký ức (chỉnh sửa file này)
├── audio/                 # Đặt file audio đã lọc nhiễu vào đây
└── memory_db/              # Tự tạo ra sau khi chạy build_database.py
```

## 7. Ghi chú quan trọng

- **Không bịa dữ liệu cho người thật đã mất** khi chưa có nguồn xác thực —
  xem lại phần trao đổi trước đó về đạo đức dữ liệu trong dự án này.
- Trường `"xac_thuc"` trong dataset dùng để app tự động cảnh báo người dùng
  biết đâu là dữ liệu demo, đâu là dữ liệu đã kiểm chứng thật.
- Model `claude-haiku-4-5-20251001` được chọn vì nhanh và rẻ, phù hợp demo.
  Khi có ngân sách, có thể đổi sang `claude-sonnet-5` trong `app.py` (biến
  `CLAUDE_MODEL`) để câu trả lời tự nhiên và sâu sắc hơn.
