import os

# Neo đường dẫn theo repo root (thư mục cha của package doctutor/), không phụ
# thuộc thư mục chạy lệnh — trùng với đường dẫn mount trong docker-compose.yml.
DB_PATH: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "chroma_db")

LLM_MODEL: str = "qwen2.5:3b"
EMBED_MODEL: str = "bge-m3"
COLLECTION_NAME: str = "documents"

CHUNK_SIZE: int = 1000      # số ký tự tối đa mỗi chunk
CHUNK_OVERLAP: int = 200    # số ký tự ngữ cảnh lặp lại giữa hai chunk liền nhau
EMBED_BATCH_SIZE: int = 32  # số chunk embed mỗi request gửi Ollama
TOP_K: int = 4              # số đoạn truy xuất cho mỗi câu hỏi
NUM_CTX: int = 8192         # context window khi gọi LLM
