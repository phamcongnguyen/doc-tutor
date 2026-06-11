# Work Plan — DocTutor

Follow the phase order. Respect dependencies (marked "Needs: ...").

---

## Phase 0 — Foundation (do first; every later phase depends on this)

### T0.1 — Write the code base as in the reference material
First step: get a working version running, exactly like the RAG pipeline in the
original material (`docs/reference.pdf`). The goal is for the whole team to
understand each step and to have a working foundation to improve on — optimization
is not the point yet. Implement the full basic flow: read PDF → chunk → create
embeddings → store in vector database → retrieve relevant passages → build the
prompt and ask the LLM, then wrap it in a Streamlit UI that allows uploading a PDF
and asking questions. This is the first draft — tasks T0.2 to T0.5 will revise the
necessary parts, so at this step just follow the material closely for clarity.
**Note:** this is not the final destination. Don't optimize early; make it work first.
**Done when:** the app runs, you can upload a PDF and get answers based on the
document content.

### T0.2 — Set up a shared repo and split files
After having a working version (T0.1): when the whole team works together and adds
many features, a single file gets messy and prone to conflicts. Set up a shared
repo (Git), split the program into core logic (`rag_core.py`) and UI (`app.py`).
Add a dependency list (`requirements.txt`) and run instructions (`README.md`).
Agree on naming and commit conventions to avoid chaos.
**Done when:** every member can clone the repo and run the app just like T0.1, but
now in clean multi-file form.

### T0.3 — Attach page numbers to each document passage
The most important foundational task. Currently when splitting the document, the
system keeps only the text and discards which page each passage came from — while
the source-citation feature (Phase 2) needs it. Modify the reading and splitting
step so each passage remembers its page, then save that info into the vector store.
Must be done early to avoid reworking many parts later.
**Done when:** for any passage the system retrieves, you can trace back its original
page number.

### T0.4 — Organize a shared data store for multiple documents
Currently each upload creates a separate, isolated store — fine for one file but it
blocks asking across multiple documents at once (Phase 2). Switch to a single shared
store, tagging each passage with which file it belongs to so you can later filter by
file. This is a data-organization decision, locked in from the start, not changed
midway.
**Done when:** uploading two different files keeps their data separate, and you can
tell which passage belongs to which file.

### T0.5 — Persistent storage and switching to a Vietnamese-capable model
Two small items. (1) Currently closing the app loses all processed data, and next
time it must be redone from scratch — make the system persist data to disk for
reuse. (2) The default model in the reference handles Vietnamese poorly — switch to
a better one (Qwen2.5 or Gemma2). Not technically hard but it greatly affects the
experience, so do it early.
**Done when:** closing and reopening doesn't require reprocessing old documents, and
Vietnamese answers are clearer and more natural.

---

## Phase 1 — Core UX

### T1.1 — Conversation memory
Currently the bot answers each question independently, with no memory of previous
ones. Let the bot "remember" what was asked and answered during the session, so the
user can ask follow-ups like "and what about what you just said?". The app already
stores chat history, so learn how to feed that history into each model call. Note:
don't send the whole history if it's too long (exceeds the context window and slows
things down) — limit it to the last few turns.
**Done when:** asking "what is YOLO?" then "is it fast?" and the bot understands
"it" = YOLO.

### T1.2 — Stream the answer as it's generated
Currently the bot waits until the whole answer is generated before showing it,
leaving the user staring at a blank screen. Make the text appear gradually, as if
being typed, for a faster and smoother feel. Learn the "streaming" mode of the model
library and how Streamlit displays streaming content. A small technical change with
a clearly better experience.
**Done when:** the answer appears gradually instead of showing up all at once after a
few seconds of waiting.

### T1.3 — Let users choose the model in the UI
Currently the model is hard-coded; changing it requires editing code. Add a dropdown
in the sidebar so users can pick the model themselves. The purpose is to easily
compare quality across models right in the UI. Make sure the model is already pulled,
and that switching it makes the next question use the newly selected model.
**Done when:** switching the model in the UI makes the next answer use that model,
with no code change.

---

## Phase 2 — Differentiating features

### T2.1 — Source citation (high priority) — Needs: T0.3
A trust-building feature: each answer states which page the information came from, so
the learner can verify it. When the bot retrieves relevant passages, also take the
accompanying page numbers and display them under the answer, e.g. "Source: page 5,
page 12". Think about a clean presentation that doesn't clutter when there are many
sources.
**Done when:** each answer can display the pages its content was drawn from.

### T2.2 — Quiz / Flashcard generator (highlight)
The most impressive feature for the learning theme: from the document, the bot
auto-generates multiple-choice questions or flashcards for self-review. Design a
separate function (a new tab or button); when clicked, the bot reads the content and
produces a list of questions/cards. The hardest part is writing the prompt that asks
the model to return a clearly structured result (question – options – correct answer),
then displaying it nicely. Experiment with the prompt several times for stable output.
**Done when:** clicking "Generate quiz" returns a set of multiple-choice questions
faithful to the content, displayed cleanly.

### T2.3 — Summarize the document / by chapter
Let users request a quick summary of the whole document or part of it. Add a
summarization function with a dedicated prompt asking the model to condense the
content. Note: long documents may exceed the context window, so for large files you
must split them, summarize each part, and combine. Start with summarizing short
documents first, then handle long ones.
**Done when:** the user requests a summary and gets one that captures the key points.

### T2.4 — Ask across multiple documents at once — Needs: T0.4
Let users upload multiple files and ask across all of them. Allow multi-file upload,
and add a way for the user to choose whether to ask across all files or only some.
Make sure the system knows which file an answer came from (pairs well with T2.1).
**Done when:** uploading 2–3 files and asking one question, the bot answers based on
exactly the selected files.

---

## Phase 3 — Advanced (do if time allows)

### T3.1 — Hybrid Search
Currently the bot only searches by meaning (semantic), which sometimes misses when
the user asks for a specific keyword or term. Add traditional keyword search too,
then combine the results of both methods to improve accuracy. Learn about keyword
search (BM25) and how to combine scores between the two methods. This task is mostly
experimentation and tuning.
**Done when:** questions containing specific terms give better results than semantic
search alone.

### T3.2 — Reranking
The goal is to filter out the most relevant passages before sending them to the
model, reducing off-topic answers. Approach: retrieve more candidate passages than
usual (e.g. 20), then use a ranking step to re-select the few most relevant. Learn
how reranking works (you may use a dedicated rerank model) and measure whether
answer quality improves. Note this adds a processing step and may slow the response
— weigh the trade-off.
**Done when:** the passages sent to the model are closer to the question, and answers
are less off-topic.

### T3.3 — Read scanned PDFs / image documents (OCR) — heaviest, do last
Currently the system can only read PDFs with real text; image/scanned PDFs yield no
content (a weakness noted in the original material). Add text recognition from images
to handle this kind of document. Learn an OCR tool (Tesseract or a vision model), and
handle documents that mix real text with image pages. This is the most complex and
time-consuming part, so only do it once the earlier phases are stable.
**Done when:** uploading a scanned PDF still lets the bot answer based on the image
content.

