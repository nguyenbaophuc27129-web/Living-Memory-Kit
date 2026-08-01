"""
app.py — Living Memory Kit demo (Streamlit)
--------------------------------------------
Giao diện chat để học sinh "trò chuyện" với một nhân chứng lịch sử,
dựa trên dữ liệu thật đã nạp vào vector database (xem build_database.py).

Chạy trước: python build_database.py   (chỉ cần 1 lần, hoặc mỗi khi đổi dataset)
Chạy app:   streamlit run app.py

Cần biến môi trường ANTHROPIC_API_KEY (xem README.md để biết cách lấy key).
"""

import os
import chromadb
import streamlit as st
from sentence_transformers import SentenceTransformer
import google.generativeai as genai

DB_PATH = "./memory_db"
COLLECTION_NAME = "living_memory"
EMBEDDING_MODEL = "bkai-foundation-models/vietnamese-bi-encoder"
GEMINI_MODEL = "gemini-2.5-flash"  # miễn phí, nhanh, đủ tốt cho demo

# ---------- Cấu hình trang ----------
st.set_page_config(page_title="Living Memory Kit", page_icon="🕯️", layout="centered")

st.markdown("""
<style>
:root{
  --lacquer:#7A2426; --gold:#C9A227; --paper:#F3EAD9;
}
.stApp{ background:var(--paper); }
[data-testid="stChatMessage"]{ border-radius:10px; }
</style>
""", unsafe_allow_html=True)


# ---------- Tải model & kết nối DB (cache để không load lại mỗi lần) ----------
@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource
def load_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_collection(COLLECTION_NAME)


@st.cache_resource
def load_gemini_client():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        st.error(
            "Chưa cấu hình GEMINI_API_KEY. "
            "Xem README.md để biết cách lấy key miễn phí tại aistudio.google.com."
        )
        st.stop()
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(GEMINI_MODEL)


embed_model = load_embedding_model()
collection = load_collection()
gemini = load_gemini_client()


# ---------- Chọn nhân chứng ----------
st.title("🕯️ Living Memory Kit")
st.caption("Trò chuyện cùng nhân chứng lịch sử — dựa trên dữ liệu đã số hóa")

all_witnesses = sorted(set(m["nhan_chung"] for m in collection.get()["metadatas"]))
witness = st.selectbox("Chọn nhân chứng bạn muốn trò chuyện:", all_witnesses)

st.divider()

# ---------- Lịch sử chat (theo từng nhân chứng) ----------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = {}
if witness not in st.session_state.chat_history:
    st.session_state.chat_history[witness] = []

for msg in st.session_state.chat_history[witness]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("audio_path") and os.path.exists(msg["audio_path"]):
            st.audio(msg["audio_path"], start_time=msg.get("audio_start", 0))
        if msg.get("nguon_note"):
            st.caption(msg["nguon_note"])


# ---------- Hàm truy vấn + trả lời ----------
def retrieve_memory(query: str, witness: str, n_results: int = 2):
    q_emb = embed_model.encode(query).tolist()
    results = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        where={"nhan_chung": witness},
    )
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    return list(zip(docs, metas))


def ask_gemini(query: str, witness: str, retrieved) -> str:
    context = "\n\n".join(
        f"[Đoạn ký ức - nguồn: {m['nguon']}]\n{d}" for d, m in retrieved
    )
    prompt = f"""Bạn đang mô phỏng lại lời kể của {witness}, một nhân chứng lịch sử,
CHỈ dựa trên các đoạn ký ức thật được cung cấp bên dưới. Đây là công cụ giáo dục lịch sử,
nên phải tuyệt đối trung thực:

- KHÔNG được bịa thêm chi tiết, số liệu, tên người hay sự kiện không có trong đoạn ký ức.
- Nếu câu hỏi không có thông tin liên quan trong đoạn ký ức, hãy trả lời trung thực rằng
  chưa có dữ liệu về phần đó, thay vì tự suy diễn.
- Xưng hô và văn phong gần gũi, như đang kể chuyện cho một học sinh nghe.

Các đoạn ký ức thật liên quan đến câu hỏi:
{context}

Câu hỏi của học sinh: {query}
"""

    response = gemini.generate_content(
        prompt,
        generation_config={"max_output_tokens": 500},
    )
    return response.text


# ---------- Ô nhập câu hỏi ----------
query = st.chat_input(f"Đặt câu hỏi cho {witness}...")

if query:
    st.session_state.chat_history[witness].append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.write(query)

    retrieved = retrieve_memory(query, witness)

    if not retrieved:
        answer = "Xin lỗi, hiện chưa có dữ liệu ký ức nào liên quan đến câu hỏi này."
        audio_meta = {}
    else:
        answer = ask_gemini(query, witness, retrieved)
        audio_meta = retrieved[0][1]  # metadata của đoạn khớp nhất

    is_verified = audio_meta.get("xac_thuc", False)
    nguon_note = (
        f"📌 Nguồn: {audio_meta.get('nguon', 'không rõ')}"
        if audio_meta
        else None
    )
    if audio_meta and not is_verified:
        nguon_note = "⚠️ DỮ LIỆU MẪU — chưa xác thực, chỉ dùng để demo kỹ thuật. " + nguon_note

    with st.chat_message("assistant"):
        st.write(answer)
        audio_path = audio_meta.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            st.audio(audio_path, start_time=audio_meta.get("audio_start", 0))
        if nguon_note:
            st.caption(nguon_note)

    st.session_state.chat_history[witness].append({
        "role": "assistant",
        "content": answer,
        "audio_path": audio_meta.get("audio_path"),
        "audio_start": audio_meta.get("audio_start", 0),
        "nguon_note": nguon_note,
    })