"""Logic hỏi đáp (tab Chat): dựng prompt từ context + lịch sử rồi stream câu trả lời.

Bước truy hồi context/cites nằm ở rag_core.retrieve (hạ tầng RAG dùng chung).
"""

import config
import prompts
import rag_core


def answer(question, chat_history, context, model = config.LLM_MODEL):
  """Sinh câu trả lời dạng stream dựa trên context đã truy hồi và lịch sử chat."""
  history = chat_history[-config.HISTORY_MESSAGES:]  # giữ 3 lượt gần nhất (mỗi lượt = user + assistant)
  messages = [
    *history,
    {"role": "user", "content": prompts.CHAT_PROMPT.format(context = context, question = question)},
  ]
  yield from rag_core.llm_chat(messages, model = model, stream = True)
