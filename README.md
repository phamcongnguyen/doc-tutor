# DocTutor

> A study-material Q&A chatbot — turn any document into your private tutor.
> (The name "DocTutor" is a working title and may change.)

DocTutor lets you upload a PDF (textbook, slides, research paper) and ask questions
in Vietnamese. The system finds the relevant passages in the document and answers
based on them, using **RAG (Retrieval Augmented Generation)**.

All AI models **run locally via Ollama** — no paid API, no constant internet needed.

## Features

- Q&A over PDF documents in Vietnamese
- Source citation (shows which page the answer came from)
- Automatic quiz / flashcard generation from the document
- Document summarization
- Conversation memory for follow-up questions
- Asking across multiple documents at once

(See the detailed roadmap in [`docs/PLAN.md`](docs/PLAN.md).)

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
   ollama pull qwen2.5:3b      # or gemma2:9b if your machine is powerful enough
   ```

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

## Project structure

```
.
├── app.py              # Streamlit UI
├── rag_core.py         # Core logic: read PDF, chunk, embed, retrieve, Q&A
├── requirements.txt    # Python dependencies
└── docs/
    ├── PLAN.md         # Work plan by phase
    ├── reference.pdf   # Original AI VIET NAM material (reference)
    └── overview.pdf    # Product overview document
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
