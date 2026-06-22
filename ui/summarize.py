import streamlit as st
import rag_core
import summarize as summarize_logic  # module logic ở gốc dự án (khác file UI này)


def render(collection, model):
  if collection.count() == 0:
    st.info("Upload và xử lý PDF trước khi tóm tắt.")
    return

  src = st.selectbox("Chọn tài liệu", rag_core.list_sources(collection), key="sum_src")
  if st.button("Tóm tắt tài liệu"):
    try:
      stream = summarize_logic.summarize(collection, src, model)
      st.write_stream(stream)
    except Exception as e:
      st.error(f"Tóm tắt lỗi, thử lại nhé ({e})")
