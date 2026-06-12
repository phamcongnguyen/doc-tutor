from __future__ import annotations

import io

import pypdf


def extract_pages(data: bytes) -> list[tuple[int, str]]:
  """Đọc PDF (bytes) và trả về list (số trang, text); trang đánh số từ 1.

  Trang scan/ảnh không trích được text sẽ trả về chuỗi rỗng.
  """
  reader = pypdf.PdfReader(io.BytesIO(data))
  return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
