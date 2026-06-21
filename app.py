import streamlit as st
import rag_core
import quiz
import summarize

MODELS = ["qwen2.5:3b", "gemma2:9b"]   # danh sách model đã pull sẵn

tab_chat, tab_quiz, tab_summarize = st.tabs(["Chat", "Quiz", "Tóm tắt"])

def format_cites(cites):
  # "a.pdf (trang 3, 7) · b.pdf (trang 2)"
  return " · ".join(
    f"{s} (trang {', '.join(map(str, pages))})" for s, pages in cites.items()
  )

# Khởi tạo Session State
st.session_state.collection = rag_core.get_collection()
for k, v in {"collection": None, "pdf_name": "", "chat_history": [], "quiz": []}.items():
  st.session_state.setdefault(k,v)

with st.sidebar:
  st.subheader("Upload tài liệu")
  model = st.selectbox("Chọn model", MODELS)
  files = st.file_uploader("Chọn file PDF", type="pdf", accept_multiple_files=True)
  if files and st.button("Xử lý PDF", use_container_width=True):
    with st.spinner("Đang xử lý..."):
      total = 0
      for file in files:
        st.session_state.collection, n = rag_core.process_pdf(file)
        total += n
        if n == 0:
          st.warning("Không trích được text — file có thể là PDF scan/ảnh.")
    st.success(f"Đã xử lý {len(files)} file — tổng {total} chunks")

  st.info(f"{len(rag_core.list_sources(st.session_state.collection))} tài liệu")
  if st.button("Xóa lịch sử chat", use_container_width=True):
    st.session_state.chat_history = []

# Tab chat
with tab_chat:
  # Giao diện người dùng (UI)
  st.set_page_config(page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
  st.title("PDF RAG Assistant: Native")

  if st.session_state.collection.count() == 0:
    st.info("Upload và xử lý PDF trước khi chat.")
  else:
    all_sources = rag_core.list_sources(st.session_state.collection)
    selected = st.multiselect("Hỏi trong tài liệu:", all_sources, default = all_sources, key="chat_sources")
    if not selected:
      st.info("Chọn ít nhất 1 tài liệu để hỏi.")

    # Lịch sử chat
    for m in st.session_state.chat_history:
      with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("cites"):
          st.caption("📄 Nguồn: " + format_cites(m["cites"]))

    # Xử lý câu hỏi vừa gửi (lưu ở pending_q bởi ô nhập ghim đáy màn hình)
    if st.session_state.get("pending_q"):
      q = st.session_state.pending_q
      st.session_state.pending_q = None
      with st.chat_message("user"):
        st.write(q)
      with st.chat_message("assistant"):
        cites = {}
        try:
          context, cites = rag_core.retrieve(q, st.session_state.collection, selected)
          stream = rag_core.rag(q, st.session_state.chat_history, context, model)
          ans = st.write_stream(stream)
          st.caption("📄 Nguồn: " + format_cites(cites))
        except Exception as e:
          ans = f"Lỗi khi gọi model `{model}` — model đã được pull chưa? ({e})"
          st.error(ans)
      st.session_state.chat_history.append({"role": "user", "content": q})
      st.session_state.chat_history.append({"role": "assistant", "content": ans, "cites": cites})

    # Ô nhập câu hỏi — đặt cuối tab Chat (dưới các tin nhắn)
    if prompt := st.chat_input("Nhập câu hỏi của bạn...", disabled=not selected):
      st.session_state.pending_q = prompt
      st.rerun()

# Tab quiz
with tab_quiz:
  if st.session_state.collection.count() == 0:
    st.info("Upload và xử lý PDF trước khi tạo quiz.")
  else:
    n = st.slider("Số câu hỏi", 3, 10, 5)
    src = st.selectbox("Chọn tài liệu", rag_core.list_sources(st.session_state.collection), key="quiz_src")
    if st.button("Tạo quiz"):
      with st.spinner("Đang tạo câu hỏi ..."):
        try:
          st.session_state.quiz = quiz.generate_quiz(
            st.session_state.collection, src,
            model = model, n_question = n,
          )
        except Exception as e:
          st.error(f"Tạo quiz lỗi, thử lại nhé ({e})")

    # Render quiz từ session_state (chạy ở mọi rerun)
    if st.session_state.quiz:
      with st.form("quiz_form"):
        choices = []
        for i, item in enumerate(st.session_state.quiz):
          st.markdown(f"**Câu {i+1}: {item['question']}**")
          # index=None để không chọn sẵn đáp án nào
          choice = st.radio("Chọn:", options=range(len(item["options"])),
                            format_func=lambda x, it=item: it["options"][x],
                            key=f"q{i}", index=None, label_visibility="collapsed"
                            )
          choices.append(choice)
        submitted = st.form_submit_button("Nộp bài")

      if submitted:
        score = 0
        for i, item in enumerate(st.session_state.quiz):
          dung = choices[i] == item["answer"]
          score += dung
          if dung:
            st.success(f"Câu {i+1}: Đúng ✅")
          else:
            dap_an = item["options"][item["answer"]]
            st.error(f"Câu {i+1}: Sai ❌ — Đáp án đúng: {dap_an}")
        st.info(f"Kết quả: {score}/{len(st.session_state.quiz)}")

# Tab summarize
with tab_summarize:
  if st.session_state.collection.count() == 0:
    st.info("Upload và xử lý PDF trước khi tóm tắt.")
  else:
    src = st.selectbox("Chọn tài liệu", rag_core.list_sources(st.session_state.collection), key="sum_src")
    if st.button("Tóm tắt tài liệu"):
      try:
        stream = summarize.summarize(st.session_state.collection, src, model)
        st.write_stream(stream)
      except Exception as e:
        st.error(f"Tóm tắt lỗi, thử lại nhé ({e})")
