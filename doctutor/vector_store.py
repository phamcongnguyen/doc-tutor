from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict, cast

import chromadb


class Chunk(TypedDict):
  """Một đoạn text đã cắt, kèm số trang gốc trong PDF."""

  text: str
  page: int


class ChunkMeta(TypedDict):
  """Metadata lưu kèm mỗi chunk trong ChromaDB."""

  page: int
  source: str


class VectorStore:
  """Bọc ChromaDB: một collection chung cho mọi file (T0.4), dữ liệu persist
  xuống đĩa để dùng lại giữa các lần chạy (T0.5).
  """

  def __init__(self, path: str, collection_name: str) -> None:
    self.client = chromadb.PersistentClient(path=path)
    self.collection = self.client.get_or_create_collection(collection_name)

  def add(self, source: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
    """Lưu các chunk của một file."""
    self.collection.add(
      ids=[f"{source}_{i}" for i in range(len(chunks))],
      documents=[c["text"] for c in chunks],
      # cast vì chroma khai báo kiểu embeddings hẹp hơn (list bất biến theo
      # phần tử), dù list[list[float]] chạy đúng
      embeddings=cast("list[Sequence[float] | Sequence[int]]", vectors),
      metadatas=[{"page": c["page"], "source": source} for c in chunks],
    )

  def delete_source(self, name: str) -> None:
    """Xóa mọi chunk của một file (tránh nhân đôi khi upload lại cùng tên)."""
    self.collection.delete(where={"source": name})

  def query(self, vectors: list[list[float]], k: int) -> list[tuple[str, ChunkMeta]]:
    """Trả về k đoạn gần nhất với vector câu hỏi: list (text, metadata)."""
    res = self.collection.query(
      query_embeddings=cast("list[Sequence[float] | Sequence[int]]", vectors), n_results=k)
    docs = cast("list[str]", res["documents"])[0]
    metas = cast("list[list[ChunkMeta]]", res["metadatas"])[0]
    return list(zip(docs, metas))

  def count(self) -> int:
    """Tổng số chunk đang có trong collection."""
    return self.collection.count()

  def sources(self) -> list[str]:
    """Danh sách tên file đã được index."""
    metas = cast("list[ChunkMeta]", self.collection.get(include=["metadatas"])["metadatas"])
    return sorted({m["source"] for m in metas})
