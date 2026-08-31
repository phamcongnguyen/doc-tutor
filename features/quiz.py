"""Logic sinh câu hỏi trắc nghiệm (tab Quiz) từ nội dung tài liệu."""

import json, re

import chromadb

import config
import prompts
import rag_core


def _parse_quiz(raw):
  # Bóc ```json ... ``` nếu model lỡ bọc, rồi lấy đúng mảng [...]
  raw = re.sub(r"```(?:json)?|```", "", raw).strip()
  start, end = raw.find("["), raw.rfind("]")
  if start == -1 or end == -1:
      raise ValueError("Model không trả về JSON hợp lệ")
  return json.loads(raw[start:end + 1])

def generate_quiz(collection: chromadb.Collection, source: str, model: str, n_question: int):
  context = rag_core.get_doc_text(collection, source, max_chars = config.QUIZ_MAX_CHARS)
  raw = rag_core.llm_chat(
    [{"role": "user", "content": prompts.QUIZ_PROMPT.format(n = n_question, context = context)}],
    model = model,
  )
  return _parse_quiz(raw)
