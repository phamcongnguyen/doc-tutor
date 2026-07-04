import streamlit as st
import rag_core
from features import quiz as quiz_logic  # logic sinh quiz (khác file UI này)


def render(collection, model):
  if collection.count() == 0:
    st.info("Upload và xử lý PDF trước khi tạo quiz.")
    return

  n = st.slider("Số câu hỏi", 3, 10, 5)
  src = st.selectbox("Chọn tài liệu", rag_core.list_sources(collection), key="quiz_src")
  if st.button("Tạo quiz"):
    with st.spinner("Đang tạo câu hỏi ..."):
      try:
        st.session_state.quiz = quiz_logic.generate_quiz(
          collection, src,
          model=model, n_question=n,
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
