from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

if TYPE_CHECKING:
  from doctutor.rag import RAGCore


def render_chat(core: RAGCore) -> None:
  """Khung chat: hiển thị lịch sử hội thoại và xử lý câu hỏi mới."""
  for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
      st.write(m["content"])

  if core.count() == 0:
    st.info("Upload và xử lý PDF trước khi chat.")
    st.chat_input("Nhập câu hỏi...", disabled=True)
    return

  q = st.chat_input("Nhập câu hỏi của bạn...")
  if not q:
    return

  with st.chat_message("user"):
    st.write(q)
  with st.chat_message("assistant"):
    with st.spinner("Đang suy nghĩ..."):
      try:
        ans = core.ask(q)
      except Exception:
        st.error("Không gọi được Ollama — kiểm tra Ollama đang chạy và "
                 "model đã được pull (`ollama list`).")
        st.stop()
      st.write(ans)
  # Chỉ lưu lịch sử khi đã có câu trả lời, để lỗi không làm lệch hội thoại
  st.session_state.chat_history += [
    {"role": "user", "content": q},
    {"role": "assistant", "content": ans},
  ]
