# DocTutor

> A study-material Q&A chatbot — turn any document into your private tutor.
> (The name "DocTutor" is a working title and may change.)

DocTutor lets you upload a PDF (textbook, slides, research paper) and ask questions
in Vietnamese. The system finds the relevant passages in the document and answers
based on them, using **RAG (Retrieval Augmented Generation)**.

All AI models **run locally via Ollama** — no paid API, no constant internet needed.

## Features

**Working now**

- Q&A over PDF documents in Vietnamese, fully local via Ollama
- Persistent index — processed documents survive app restarts
- Page & source metadata stored for every passage (groundwork for citations)

**Planned** (see the detailed roadmap in [`docs/PLAN.md`](docs/PLAN.md))

- Source citation (shows which page the answer came from)
- Automatic quiz / flashcard generation from the document
- Document summarization
- Conversation memory for follow-up questions
- Asking across multiple documents at once

## Tech stack

| Component       | Technology                   |
|-----------------|------------------------------|
| UI              | Streamlit                    |
| PDF reading     | pypdf                        |
| Vector Database | ChromaDB                     |
| Embedding       | BGE-M3 (via Ollama)          |
| LLM             | Qwen2.5 / Gemma2 (via Ollama)|

## Requirements

- Python 3.9 or higher
- [Ollama](https://ollama.com) installed and running

## Installation

1. Install Ollama from https://ollama.com, then pull the models:

   ```bash
   ollama pull bge-m3
   ollama pull qwen2.5:3b
   ```

   The LLM defaults to `qwen2.5:3b`. To use another model (e.g. `gemma2:9b` on a
   powerful machine), pull it and change `LLM_MODEL` in `doctutor/config.py` — a
   model picker in the UI is planned (T1.3).

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Running the app

Make sure Ollama is running (check: `ollama list`), then:

```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

### Running with Docker

The app runs in Docker; Ollama stays **native on the host** (Docker on macOS has
no Metal GPU access). The container reaches it via `host.docker.internal`.

```bash
docker compose -f docker/docker-compose.yml up --build
```

For IDE autocomplete without running the app locally, create the `vendor` venv:

```bash
./macos-script.sh
source vendor/bin/activate
```

## Project structure

```
.
├── app.py                  # Entry point: streamlit run app.py
├── doctutor/               # Application package
│   ├── config.py           # Defaults: models, DB path, chunking & retrieval params
│   ├── pdf_loader.py       # Read PDF -> (page number, text) pairs
│   ├── chunker.py          # Chunker: split text by sentence with overlap
│   ├── embedder.py         # OllamaEmbedder: batched embeddings via Ollama
│   ├── vector_store.py     # VectorStore: ChromaDB wrapper (add/query/delete)
│   ├── llm.py              # OllamaLLM: prompt template + chat call
│   ├── rag.py              # RAGCore: composes the pipeline end-to-end
│   └── ui/
│       ├── sidebar.py      # Upload & index status sidebar
│       └── chat.py         # Chat history + Q&A flow
├── requirements.txt        # Python dependencies
├── macos-script.sh         # Create "vendor" venv for IDE autocomplete (app runs in Docker)
├── chroma_db/              # Created at runtime — persisted vector index (gitignored)
├── docker/
│   ├── Dockerfile
│   ├── Dockerfile.dockerignore
│   └── docker-compose.yml
└── docs/
    ├── PLAN.md             # Work plan by phase
    ├── reference.pdf       # Original AI VIET NAM material (reference)
    └── overview.pdf        # Product overview document
```

## Usage

1. Open the app and upload a PDF from the sidebar.
2. Click process so the system reads and indexes the document.
3. Ask questions in the chat box — the bot answers based on the document content.

## Team members

<!-- Fill in member names and assignments here -->

| Member | Responsible for |
|--------|-----------------|
|        |                 |
