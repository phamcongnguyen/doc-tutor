import streamlit as st
import rag_core

MODELS = ["qwen2.5:3b", "gemma2:9b"]   # danh sách model đã pull sẵn

# Khởi tạo Session State
st.session_state.collection = rag_core.get_collection()
for k, v in {"collection": None, "pdf_name": "", "chat_history": []}.items():
  st.session_state.setdefault(k,v)

# Giao diện người dùng (UI)
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("PDF RAG Assistant: Native")

with st.sidebar:
  st.subheader("Upload tài liệu")
  model = st.selectbox("Chọn model", MODELS)
  f = st.file_uploader("Chọn file PDF", type="pdf")
  if f and st.button("Xử lý PDF", use_container_width=True):
    with st.spinner("Đang xử lý..."):
      st.session_state.collection, n = rag_core.process_pdf(f)
      st.session_state.pdf_name = f.name
      st.session_state.chat_history = []
    if n == 0:
      st.warning("Không trích được text — file có thể là PDF scan/ảnh.")
    else:
      st.success(f"{n} chunks")
  st.info(f" {st.session_state.pdf_name}" if st.session_state.pdf_name else " Chưa có tài liệu")
  if st.button("Xóa lịch sử chat", use_container_width=True):
    st.session_state.chat_history = []

for m in st.session_state.chat_history:
  with st.chat_message(m["role"]):
    st.write(m["content"])

if st.session_state.collection.count() == 0:
  st.info("Upload và xử lý PDF trước khi chat.")
  st.chat_input("Nhập câu hỏi...", disabled=True)
else:
  q = st.chat_input("Nhập câu hỏi của bạn...")
  if q:
    with st.chat_message("user"):
      st.write(q)
    with st.chat_message("assistant"):
      try:  
        stream = rag_core.rag(q, st.session_state.collection, st.session_state.chat_history, model)
        ans = st.write_stream(stream)
      except Exception as e:
        ans = f"Lỗi khi gọi model `{model}` — model đã được pull chưa? ({e})"
        st.error(ans)
    st.session_state.chat_history.append({"role": "user", "content": q})
    st.session_state.chat_history.append({"role": "assistant", "content": ans})