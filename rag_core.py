import tempfile, os, time
import pypdf
import chromadb
import ollama

client = chromadb.PersistentClient(path="chroma_db")

LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"
DOCUMENTS = "documents"

PROMPT = """Bạn là trợ lý hỏi đáp. Dùng các đoạn ngữ cảnh dưới đây để trả lời câu hỏi.
Nếu ngữ cảnh không có thông tin, hãy nói là bạn không biết, đừng bịa.
Trả lời ngắn gọn, chính xác, bằng tiếng Việt.

Ngữ cảnh: {context}

Câu hỏi: {question}
Trả lời:"""

# Các hàm xử lý (core functions)
def embed(texts):
  """Chuyển text thành vector embedding."""
  return ollama.embed(model = EMBED_MODEL, input = texts)["embeddings"]

def chunk_text(text, size = 1000, overlap = 200):
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
  col = client.get_or_create_collection(DOCUMENTS)
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

def rag(question, collection, chat_history, model = LLM_MODEL, k = 4):
  """Hàm RAG: tìm context và hỏi LLM."""
  res = collection.query(query_embeddings = embed([question]), n_results = k)
  context = "\n\n".join(res["documents"][0])
  history = chat_history[-6:] # giữ 3 lượt gần nhất (mỗi lượt = user + assistant)
  stream = ollama.chat(
    model = model,
    messages = [*history, {"role": "user", "content": PROMPT.format(context = context, question = question)}],
    options = {"temperature": 0},
    stream = True,
  )
  for chunk in stream:
    yield chunk["message"]["content"]

def get_collection():
  return client.get_or_create_collection(DOCUMENTS)