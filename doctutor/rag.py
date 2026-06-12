from __future__ import annotations

from typing import Protocol

from . import config
from .chunker import Chunker
from .embedder import OllamaEmbedder, ProgressCallback
from .llm import OllamaLLM
from .pdf_loader import extract_pages
from .vector_store import Chunk, VectorStore


class UploadedFileLike(Protocol):
  """Phần giao diện pipeline cần từ một file upload: tên file và nội dung
  bytes. Khớp với UploadedFile của Streamlit mà core không phải import
  streamlit."""

  name: str

  def getvalue(self) -> bytes: ...


class RAGCore:
  """Pipeline RAG: đọc PDF -> chunk -> embed -> lưu ChromaDB -> truy xuất -> hỏi LLM.

  Chỉ ghép các thành phần độc lập (pdf_loader, Chunker, OllamaEmbedder,
  VectorStore, OllamaLLM) lại với nhau; giá trị mặc định lấy từ `config`
  để chỉnh ở một chỗ.
  """

  def __init__(self, db_path: str | None = None, llm_model: str = config.LLM_MODEL,
               embed_model: str = config.EMBED_MODEL,
               collection_name: str = config.COLLECTION_NAME) -> None:
    self.chunker = Chunker(config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    self.embedder = OllamaEmbedder(embed_model, config.EMBED_BATCH_SIZE)
    self.store = VectorStore(db_path or config.DB_PATH, collection_name)
    self.llm = OllamaLLM(llm_model, config.NUM_CTX)

  def process_pdf(self, uploaded_file: UploadedFileLike,
                  on_progress: ProgressCallback | None = None) -> int:
    """Đọc PDF, cắt nhỏ, embed và lưu vào ChromaDB. Trả về số chunk đã lưu."""
    chunks: list[Chunk] = []
    for page_no, text in extract_pages(uploaded_file.getvalue()):
      for c in self.chunker.split(text):
        chunks.append({"text": c, "page": page_no})

    # PDF scan/rỗng không trích được text: vẫn xóa bản cũ cùng tên (nếu có)
    # để tránh trả lời theo nội dung lỗi thời, rồi trả về sớm.
    if not chunks:
      self.store.delete_source(uploaded_file.name)
      return 0

    # Embed xong mới xóa bản cũ: nếu embed lỗi (Ollama chưa chạy, chưa pull
    # model...) thì dữ liệu đã index trước đó vẫn còn nguyên.
    vectors = self.embedder.embed([c["text"] for c in chunks], on_progress=on_progress)
    self.store.delete_source(uploaded_file.name)
    self.store.add(uploaded_file.name, chunks, vectors)
    return len(chunks)

  def ask(self, question: str, k: int = config.TOP_K) -> str:
    """Truy xuất các đoạn liên quan và hỏi LLM."""
    hits = self.store.query(self.embedder.embed([question]), k)
    # Gắn nhãn nguồn cho từng đoạn để model không trộn các đoạn rời rạc làm một
    context = "\n\n".join(
      f"[Đoạn {i + 1} — {m['source']}, trang {m['page']}]\n{d}"
      for i, (d, m) in enumerate(hits)
    )
    return self.llm.answer(question, context)

  def count(self) -> int:
    """Tổng số chunk đang có trong collection."""
    return self.store.count()

  def sources(self) -> list[str]:
    """Danh sách tên file đã được index."""
    return self.store.sources()
