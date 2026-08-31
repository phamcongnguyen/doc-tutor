MODELS = ["qwen2.5:3b", "gemma2:9b"]   # danh sách model đã pull sẵn
LLM_MODEL = "qwen2.5:3b"
EMBED_MODEL = "bge-m3"

# --- Lưu trữ ---
CHROMA_PATH = "chroma_db"   # thư mục ChromaDB persist xuống đĩa
DOCUMENTS = "documents"     # tên collection dùng chung cho mọi file

# --- Chunk & embed ---
DEFAULT_TEXT_SIZE = 1000
DEFAULT_TEXT_OVERLAP = 200
EMBED_BATCH = 64           # số chunk gửi mỗi lần gọi embedding (tránh 1 request quá lớn)

# --- Đọc cả tài liệu (quiz / tóm tắt) ---
QUIZ_MAX_CHARS = 6000             # cắt bớt tài liệu trước khi cho model nhỏ ra đề
SUMMARY_SINGLE_PASS_CHARS = 8000  # dài hơn ngưỡng này thì tóm tắt theo từng phần rồi gộp
SUMMARY_GROUP_CHARS = 6000        # kích thước mỗi phần khi chia tài liệu dài để tóm tắt

# --- Truy hồi (retrieve) ---
RETRIEVE_K = 4             # số chunk lấy ra mỗi lần retrieve
HISTORY_MESSAGES = 6       # số message lịch sử giữ lại (3 lượt user+assistant)
RETRIEVE_CANDIDATES = 20   # số ứng viên lấy ra từ mỗi phương pháp trước khi gộp
RRF_K = 60                 # hằng số làm mượt của Reciprocal Rank Fusion
RERANK_MODEL = "BAAI/bge-reranker-v2-m3"  # cross-encoder đa ngữ, cùng họ BGE-M3
USE_RERANK = True                          # cờ bật/tắt để so sánh & phòng khi demo lỗi
