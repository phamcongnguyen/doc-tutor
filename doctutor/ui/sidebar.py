from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
  from streamlit.runtime.uploaded_file_manager import UploadedFile

  from doctutor.rag import RAGCore


def render_sidebar(core: RAGCore) -> None:
  """Sidebar: upload + xử lý PDF, trạng thái index, nút xóa lịch sử chat."""
  with st.sidebar:
    st.subheader("Upload tài liệu")
    f = st.file_uploader("Chọn file PDF", type="pdf")
    if f and st.button("Xử lý PDF", use_container_width=True):
      _process_pdf(core, f)

    # Hiển thị theo dữ liệu thật trong collection (vẫn đúng sau khi restart app)
    docs = core.sources()
    st.info("Đã index: " + ", ".join(docs) if docs else "Chưa có tài liệu")
    if st.button("Xóa lịch sử chat", use_container_width=True):
      st.session_state.chat_history = []


def _process_pdf(core: RAGCore, f: UploadedFile) -> None:
  bar = st.progress(0.0, text="Đang đọc PDF...")
  try:
    n = core.process_pdf(
      f,
      on_progress=lambda done, total: bar.progress(
        done / total, text=f"Đang embed {done}/{total} chunks..."),
    )
  except Exception:
    st.error("Xử lý thất bại — kiểm tra Ollama đang chạy, model đã pull "
             "(`ollama list`) và file PDF không hỏng/khóa mật khẩu.")
  else:
    st.session_state.chat_history = []
    if n == 0:
      st.warning("Không trích được text — file có thể là PDF scan/ảnh.")
    else:
      st.success(f"{n} chunks")
  finally:
    bar.empty()
