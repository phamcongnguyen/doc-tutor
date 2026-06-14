from __future__ import annotations

import re


class Chunker:
  """Cắt text thành các chunk tối đa `size` ký tự theo ranh giới câu.

  pypdf xuống dòng theo dòng hiển thị chứ không theo đoạn văn, nên nối các
  dòng lại thành văn bản liền rồi mới cắt theo câu. Overlap lấy nguyên các
  câu cuối của chunk trước (không cắt giữa từ).
  """

  def __init__(self, size: int, overlap: int) -> None:
    self.size = size
    self.overlap = overlap

  def split(self, text: str) -> list[str]:
    flat = re.sub(r"\s*\n\s*", " ", text).strip()
    sents = [s for s in re.split(r"(?<=[.!?…])\s+", flat) if s]

    # Câu dài hơn size (vd: bảng biểu, text không có dấu câu) thì cắt cứng
    pieces: list[str] = []
    for s in sents:
      while len(s) > self.size:
        pieces.append(s[:self.size])
        s = s[self.size - self.overlap:]
      if s:
        pieces.append(s)

    chunks: list[str] = []
    cur: list[str] = []
    cur_len = 0
    for p in pieces:
      if cur and cur_len + len(p) + 1 > self.size:
        chunks.append(" ".join(cur))
        # Giữ lại ~overlap ký tự câu cuối làm ngữ cảnh nối sang chunk sau
        kept: list[str] = []
        kept_len = 0
        for prev in reversed(cur):
          if kept_len + len(prev) + 1 > self.overlap:
            break
          kept.insert(0, prev)
          kept_len += len(prev) + 1
        # Bảo đảm chunk mới (phần overlap + câu hiện tại) không vượt size
        while kept and kept_len + len(p) + 1 > self.size:
          kept_len -= len(kept.pop(0)) + 1
        cur, cur_len = kept, kept_len
      cur.append(p)
      cur_len += len(p) + 1
    if cur:
      chunks.append(" ".join(cur))
    return chunks
