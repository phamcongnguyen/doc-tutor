import streamlit as st

from doctutor import RAGCore
from doctutor.ui.chat import render_chat
from doctutor.ui.sidebar import render_sidebar


# Khởi tạo pipeline một lần cho cả server, dùng lại qua các lần rerun
@st.cache_resource
def get_core() -> RAGCore:
  return RAGCore()


core = get_core()
st.session_state.setdefault("chat_history", [])

st.set_page_config(page_title="PDF RAG Chatbot", layout="wide", initial_sidebar_state="expanded")
st.title("PDF RAG Assistant: Native")

render_sidebar(core)
render_chat(core)
