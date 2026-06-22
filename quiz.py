import chromadb
import json, re

import config
import rag_core

QUIZ_PROMPT = """Bạn là giáo viên ra đề. Dựa HOÀN TOÀN vào nội dung tài liệu dưới đây, hãy tạo {n} câu hỏi trắc nghiệm tiếng Việt để giúp học viên tự ôn tập.

Yêu cầu:
- Mỗi câu hỏi có đúng 4 lựa chọn, chỉ 1 đáp án đúng.
- Câu hỏi và đáp án phải bám sát nội dung tài liệu, KHÔNG bịa thông tin ngoài tài liệu.
- Các lựa chọn sai phải hợp lý (gây nhiễu), không quá lố.
- Đa dạng độ khó và bao quát nhiều phần của tài liệu.

Chỉ trả về DUY NHẤT một mảng JSON, KHÔNG kèm giải thích, KHÔNG dùng markdown, KHÔNG có chữ nào trước/sau. Định dạng chính xác:
[
  {{
    "question": "Nội dung câu hỏi?",
    "options": ["Lựa chọn A", "Lựa chọn B", "Lựa chọn C", "Lựa chọn D"],
    "answer": 0
  }}
]
Trong đó "answer" là chỉ số (0-3) của đáp án đúng trong mảng "options".

Nội dung tài liệu:
{context}
"""

def _parse_quiz(raw):
  # Bóc ```json ... ``` nếu model lỡ bọc, rồi lấy đúng mảng [...]
  raw = re.sub(r"```(?:json)?|```", "", raw).strip()
  start, end = raw.find("["), raw.rfind("]")
  if start == -1 or end == -1:
      raise ValueError("Model không trả về JSON hợp lệ")
  return json.loads(raw[start:end + 1])

def get_doc_text(collection: chromadb.Collection, source: str, max_chars = config.MAX_CHARS) -> str:
  # Gộp mọi chunk của file rồi cắt bớt nếu quá dài để khỏi quá tải model nhỏ
  context = "\n\n".join(rag_core.get_doc_chunks(collection, source))
  return context[:max_chars]

def generate_quiz(collection: chromadb.Collection, source: str, model: str, n_question: int):
  context = get_doc_text(collection = collection, source = source)
  raw = rag_core.llm_chat(
    [{"role": "user", "content": QUIZ_PROMPT.format(n = n_question, context = context)}],
    model = model,
  )
  return _parse_quiz(raw)