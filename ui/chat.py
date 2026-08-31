import streamlit as st
import rag_core
from features import chat as chat_logic  # logic sinh câu trả lời (khác file UI này)


def _format_cites(cites):
  # "a.pdf (trang 3, 7) · b.pdf (trang 2)"
  return " · ".join(
    f"{s} (trang {', '.join(map(str, pages))})" for s, pages in cites.items()
  )


def render(collection, model):
  st.title("PDF RAG Assistant: Native")

  if collection.count() == 0:
    st.info("Upload và xử lý PDF trước khi chat.")
    return

  all_sources = rag_core.list_sources(collection)
  selected = st.multiselect("Hỏi trong tài liệu:", all_sources, default=all_sources, key="chat_sources")
  if not selected:
    st.info("Chọn ít nhất 1 tài liệu để hỏi.")

  # Lịch sử chat
  for m in st.session_state.chat_history:
    with st.chat_message(m["role"]):
      st.write(m["content"])
      if m.get("cites"):
        st.caption("📄 Nguồn: " + _format_cites(m["cites"]))

  # Xử lý câu hỏi vừa gửi (lưu ở pending_q bởi ô nhập ghim đáy màn hình)
  if st.session_state.get("pending_q"):
    q = st.session_state.pending_q
    st.session_state.pending_q = None
    with st.chat_message("user"):
      st.write(q)
    with st.chat_message("assistant"):
      cites = {}
      try:
        context, cites = rag_core.retrieve(q, collection, selected)
        stream = chat_logic.answer(q, st.session_state.chat_history, context, model)
        ans = st.write_stream(stream)
        st.caption("📄 Nguồn: " + _format_cites(cites))
      except Exception as e:
        ans = f"Lỗi khi gọi model `{model}` — model đã được pull chưa? ({e})"
        st.error(ans)
    st.session_state.chat_history.append({"role": "user", "content": q})
    st.session_state.chat_history.append({"role": "assistant", "content": ans, "cites": cites})

  # Ô nhập câu hỏi — đặt cuối tab Chat (dưới các tin nhắn)
  if prompt := st.chat_input("Nhập câu hỏi của bạn...", disabled=not selected):
    st.session_state.pending_q = prompt
    st.rerun()
