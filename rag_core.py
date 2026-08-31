"""Hạ tầng RAG dùng chung: đọc PDF, chunk, embed, lưu/đọc ChromaDB, truy hồi.

Logic riêng của từng tính năng (chat / quiz / tóm tắt) nằm trong package `features/`.
"""

import tempfile, os
import pypdf
import chromadb
import ollama
import config
from rank_bm25 import BM25Okapi
import numpy as np

client = chromadb.PersistentClient(path = config.CHROMA_PATH)

_reranker = None

# Các hàm xử lý (core functions)
def embed(texts):
  """Chuyển danh sách text thành vector embedding.

  Chia thành batch (config.EMBED_BATCH) để tránh gửi 1 request quá lớn khi PDF dài."""
  out = []
  for i in range(0, len(texts), config.EMBED_BATCH):
    batch = texts[i:i + config.EMBED_BATCH]
    out.extend(ollama.embed(model = config.EMBED_MODEL, input = batch)["embeddings"])
  return out

def llm_chat(messages, model = config.LLM_MODEL, stream = False):
  """Gọi LLM qua Ollama với temperature=0 (mọi tác vụ đều cần kết quả ổn định).

  stream=False -> trả về cả câu trả lời dạng chuỗi.
  stream=True  -> trả về generator yield từng mảnh text (cho st.write_stream)."""
  resp = ollama.chat(
    model = model,
    messages = messages,
    options = {"temperature": 0},
    stream = stream,
  )
  if stream:
    return (chunk["message"]["content"] for chunk in resp)
  return resp["message"]["content"]

def chunk_text(text, size = config.DEFAULT_TEXT_SIZE, overlap = config.DEFAULT_TEXT_OVERLAP):
  """Cắt text thành các chunk nhỏ."""
  paras = [p.strip() for p in text.split("\n") if p.strip()]
  chunks, cur = [], ""
  for p in paras:
    if len(cur) + len(p) + 1 <= size:
      cur += p + "\n"
    else:
      if cur:
        chunks.append(cur.strip())
      cur = (cur[-overlap:] + p + "\n") if overlap else (p + "\n")
  if cur.strip():
    chunks.append(cur.strip())
  return chunks

def process_pdf(uploaded_file):
  """Đọc PDF, cắt nhỏ, tạo embedding và lưu vào ChromaDB."""
  # Lưu file upload thành file tạm
  with tempfile.NamedTemporaryFile(delete = False, suffix = ".pdf") as tmp:
    tmp.write(uploaded_file.getvalue())
    path = tmp.name

  # Đọc nội dung PDF
  pages = pypdf.PdfReader(path).pages
  os.unlink(path) # Xóa file tạm

  # Cắt nhỏ và lưu vào ChromaDB
  chunks = []
  for i, page in enumerate(pages, start=1):
    page_text = page.extract_text() or ""
    for c in chunk_text(page_text):
      chunks.append({"text": c, "page": i})

  # Dùng 1 collection cố định cho mọi file
  col = client.get_or_create_collection(config.DOCUMENTS)
  # PDF scan/rỗng không trích được text -> không có gì để lưu.
  # Trả về sớm để tránh lỗi embed([]) / col.add([]) khi danh sách chunk rỗng.
  if not chunks:
    return col, 0
  # Tránh nhân đôi khi upload lại cùng file, xoá chunk cũ của file đó trước khi add
  col.delete(where={"source": uploaded_file.name})
  col.add(
    ids=[f"{uploaded_file.name}_{i}" for i in range(len(chunks))],
    documents=[c["text"] for c in chunks],
    embeddings=embed([c["text"] for c in chunks]),
    metadatas=[{"page": c["page"], "source": uploaded_file.name} for c in chunks],
  )
  return col, len(chunks)

def retrieve(question: str, collection: chromadb.Collection, sources, k = config.RETRIEVE_K):
  # 1. Lấy toàn bộ chunk của các file đang chọn (kèm embeddings + metadata)
  data = collection.get(
    where={"source": {"$in": sources}},
    include=["documents", "metadatas", "embeddings"],
  )
  docs, metas = data["documents"], data["metadatas"]
  if not docs:
    return ("", {})

  # 2a. Semantic: xếp hạng theo khoảng cách cosine tới câu hỏi
  qvec = embed([question])[0]
  sem_order = _semantic_rank(qvec, data["embeddings"])  # index sắp theo gần nhất

  # 2b. Keyword: xếp hạng BM25
  bm_order = _bm25_rank(question, docs)

  # 3. Gộp bằng RRF
  scores = {}
  for rank, i in enumerate(sem_order[:config.RETRIEVE_CANDIDATES]):
    scores[i] = scores.get(i, 0) + 1 / (config.RRF_K + rank)
  for rank, i in enumerate(bm_order[:config.RETRIEVE_CANDIDATES]):
    scores[i] = scores.get(i, 0) + 1 / (config.RRF_K + rank)

  ranked = sorted(scores, key=lambda i: scores[i], reverse=True)
  if config.USE_RERANK and len(ranked) > k:
    ranked = _rerank(question, ranked, docs)  # cross-encoder chấm lại toàn bộ ứng viên RRF
  top = ranked[:k]

  # 4. Dựng context + cites như cũ
  context = "\n\n".join(docs[i] for i in top)
  by_source = {}
  for i in top:
    by_source.setdefault(metas[i]["source"], set()).add(metas[i]["page"])
  cites = {s: sorted(p) for s, p in by_source.items()}
  return (context, cites)

def get_collection() -> chromadb.Collection:
  return client.get_or_create_collection(config.DOCUMENTS)

def list_sources(collection: chromadb.Collection):
  # Chỉ cần metadata để liệt kê tên file -> không kéo cả documents về cho nhẹ
  metas = collection.get(include=["metadatas"])["metadatas"]
  return sorted({m["source"] for m in metas})

def get_doc_chunks(collection: chromadb.Collection, source: str):
  """Lấy toàn bộ chunk thuộc đúng 1 file (không tìm kiếm ngữ nghĩa).

  Dùng chung cho quiz và summarize — những tác vụ cần đọc cả tài liệu."""
  return collection.get(where = {"source": source})["documents"]

def get_doc_text(collection: chromadb.Collection, source: str, max_chars = None) -> str:
  """Gộp mọi chunk của 1 file thành một chuỗi; cắt bớt nếu vượt max_chars."""
  text = "\n\n".join(get_doc_chunks(collection, source))
  return text[:max_chars] if max_chars else text

def _tokenize(text: str):
  # Đủ dùng cho tiếng Việt giai đoạn đầu; có thể nâng cấp pyvi/underthesea sau
  return text.lower().split()

def _bm25_rank(question: str, docs: list[str]):
  """Trả về danh sách index của docs, sắp theo điểm BM25 giảm dần."""
  bm25 = BM25Okapi([_tokenize(doc) for doc in docs])
  scores = bm25.get_scores(_tokenize(question))
  return sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)

def _semantic_rank(qvec, embeddings):
  M = np.array(embeddings)
  q = np.array(qvec)
  # bge-m3 không tự chuẩn hóa -> chuẩn hóa để cosine = tích vô hướng
  M = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-10)
  q = q / (np.linalg.norm(q) + 1e-10)
  sims = M @ q
  return sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)

def _get_reranker():
  """Tải cross-encoder 1 lần rồi tái dùng (lazy, tránh chậm lúc khởi động)."""
  global _reranker
  if _reranker is None:
    from sentence_transformers import CrossEncoder
    _reranker = CrossEncoder(config.RERANK_MODEL)
  return _reranker

def _rerank(question, candidates, docs):
  """candidates: list index vào docs. Chấm lại bằng cross-encoder, trả index sắp theo điểm giảm."""
  model = _get_reranker()
  # Cross-encoder đọc CẶP (câu hỏi, đoạn) cùng lúc -> hiểu liên quan sâu hơn embedding
  pairs = [(question, docs[i]) for i in candidates]
  scores = model.predict(pairs)
  return [i for _, i in sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)]
