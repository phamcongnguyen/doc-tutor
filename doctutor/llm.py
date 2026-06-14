from __future__ import annotations

from typing import ClassVar

import ollama


class OllamaLLM:
  """Gọi LLM qua Ollama để trả lời câu hỏi dựa trên ngữ cảnh đã truy xuất."""

  PROMPT: ClassVar[str] = """Bạn là trợ lý hỏi đáp. Dưới đây là các đoạn trích rời nhau từ tài liệu,
mỗi đoạn có ghi tên file và số trang. Chỉ dùng thông tin trong các đoạn trích để trả lời câu hỏi.
Nếu các đoạn trích không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

{context}

Câu hỏi: {question}
Trả lời:"""

  def __init__(self, model: str, num_ctx: int) -> None:
    self.model = model
    self.num_ctx = num_ctx

  def answer(self, question: str, context: str) -> str:
    resp = ollama.chat(
      model=self.model,
      messages=[{"role": "user", "content": self.PROMPT.format(context=context, question=question)}],
      # num_ctx đủ lớn để Ollama không âm thầm cắt bớt phần đầu prompt
      # (instruction + ngữ cảnh) khi vượt context mặc định
      options={"temperature": 0, "num_ctx": self.num_ctx},
    )
    return resp["message"]["content"] or ""
