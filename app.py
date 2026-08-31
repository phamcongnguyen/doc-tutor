import streamlit as st

# set_page_config phải là lệnh Streamlit ĐẦU TIÊN, gọi đúng 1 lần.
st.set_page_config(page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded")

import config
import rag_core
from ui import chat, quiz, summarize

# Khởi tạo Session State
st.session_state.collection = rag_core.get_collection()
for k, v in {"pdf_name": "", "chat_history": [], "quiz": []}.items():
  st.session_state.setdefault(k, v)

# Sidebar: chọn model + upload tài liệu (dùng chung cho mọi tab)
with st.sidebar:
  st.subheader("Upload tài liệu")
  model = st.selectbox("Chọn model", config.MODELS)
  files = st.file_uploader("Chọn file PDF", type="pdf", accept_multiple_files=True)
  if files and st.button("Xử lý PDF", use_container_width=True):
    with st.spinner("Đang xử lý..."):
      total = 0
      for file in files:
        # Bọc từng file: PDF hỏng/mã hoá không được làm sập cả app hay chặn các file còn lại
        try:
          st.session_state.collection, n = rag_core.process_pdf(file)
        except Exception as e:
          st.error(f"Không xử lý được `{file.name}` — file có thể hỏng/mã hoá. ({e})")
          continue
        total += n
        if n == 0:
          st.warning(f"`{file.name}`: không trích được text — có thể là PDF scan/ảnh.")
    st.success(f"Đã xử lý xong — tổng {total} chunks")

  st.info(f"{len(rag_core.list_sources(st.session_state.collection))} tài liệu")
  if st.button("Xóa lịch sử chat", use_container_width=True):
    st.session_state.chat_history = []

# Mỗi tab ủy quyền render cho module UI tương ứng trong ui/
tab_chat, tab_quiz, tab_summarize = st.tabs(["Chat", "Quiz", "Tóm tắt"])
with tab_chat:
  chat.render(st.session_state.collection, model)
with tab_quiz:
  quiz.render(st.session_state.collection, model)
with tab_summarize:
  summarize.render(st.session_state.collection, model)
