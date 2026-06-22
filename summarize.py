import chromadb
import config
import rag_core

SUMMARY_PROMPT = """Bạn là trợ lý tóm tắt tài liệu học tập. Hãy tóm tắt nội dung dưới đây bằng tiếng Việt.

Yêu cầu:
- Nêu các ý chính dưới dạng gạch đầu dòng, ngắn gọn, dễ hiểu.
- Bám sát nội dung tài liệu, KHÔNG thêm thông tin ngoài tài liệu, KHÔNG bịa.
- Nếu tài liệu có nhiều phần/chủ đề, nhóm các ý theo từng phần.
- Bắt đầu bằng 1 câu nêu tài liệu nói về cái gì, rồi mới liệt kê ý chính.

Nội dung tài liệu:
{context}

Bản tóm tắt:"""

REDUCE_PROMPT = """Dưới đây là các bản tóm tắt của từng phần trong cùng một tài liệu.
Hãy gộp chúng thành MỘT bản tóm tắt tổng hợp bằng tiếng Việt, mạch lạc, không lặp ý, giữ lại mọi ý chính quan trọng. Trình bày dạng gạch đầu dòng.

Các bản tóm tắt từng phần:
{context}

Bản tóm tắt tổng hợp:"""

def _group_chunks(chunks, max_chars = config.MAX_CHARS):
  groups, cur = [], ""
  for char in chunks:
    if len(cur) + len(char) > max_chars and cur:
      groups.append(cur)
      cur = ""
    cur += char + "\n\n"
  if cur:
    groups.append(cur)
  return groups

def _summarize_one(text, model):
  return rag_core.llm_chat(
    [{"role": "user", "content": SUMMARY_PROMPT.format(context=text)}],
    model=model,
  )

def summarize(collection: chromadb.Collection, source: str, model: str, max_single = config.MAX_SINGLE):
  chunks = rag_core.get_doc_chunks(collection, source)
  full = "\n\n".join(chunks) 
  if len(full) <= max_single:
    prompt = SUMMARY_PROMPT.format(context = full)
  else:
    groups = _group_chunks(chunks)
    partials = [_summarize_one(group, model) for group in groups]
    prompt = REDUCE_PROMPT.format(context="\n\n".join(partials))

  yield from rag_core.llm_chat(
    [{"role": "user", "content": prompt}],
    model=model,
    stream=True,
  )
