import tempfile, os
import pypdf
import chromadb
import ollama
import config

client = chromadb.PersistentClient(path = "chroma_db")

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}
Trả lời:"""

# Các hàm xử lý (core functions)
def embed(texts):
  """Chuyển text thành vector embedding."""
  return ollama.embed(model = config.EMBED_MODEL, input = texts)["embeddings"]

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
  res = collection.query(
    query_embeddings = embed([question]),
    n_results = k,
    where={"source": {"$in": sources}})
  context = "\n\n".join(res["documents"][0])
  # Gom số trang theo từng file -> {"a.pdf": [3, 7], "b.pdf": [2]}
  by_source = {}
  for metadata in res["metadatas"][0]:
    by_source.setdefault(metadata["source"], set()).add(metadata["page"])
  cites = {s: sorted(p) for s, p in by_source.items()}
  return (context, cites)

def rag(
    question,
    chat_history,
    context,
    model = config.LLM_MODEL,
  ):
  """Hàm RAG: tìm context và hỏi LLM."""
  history = chat_history[-config.HISTORY_MESSAGES:] # giữ 3 lượt gần nhất (mỗi lượt = user + assistant)
  messages = [*history, {"role": "user", "content": PROMPT.format(context = context, question = question)}]
  yield from llm_chat(messages, model = model, stream = True)

def get_collection() -> chromadb.Collection:
  return client.get_or_create_collection(config.DOCUMENTS)

def list_sources(collection: chromadb.Collection):
  metas = collection.get()["metadatas"]
  return sorted({m["source"] for m in metas})

def get_doc_chunks(collection: chromadb.Collection, source: str):
  """Lấy toàn bộ chunk thuộc đúng 1 file (không tìm kiếm ngữ nghĩa).

  Dùng chung cho quiz và summarize — những tác vụ cần đọc cả tài liệu."""
  return collection.get(where = {"source": source})["documents"]