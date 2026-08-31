"""Logic tóm tắt tài liệu (tab Tóm tắt).

Tài liệu ngắn: tóm tắt 1 lần. Tài liệu dài: chia phần, tóm tắt từng phần rồi gộp (map-reduce).
"""

import chromadb

import config
import prompts
import rag_core


def _group_chunks(chunks, max_chars = config.SUMMARY_GROUP_CHARS):
  groups, cur = [], ""
  for chunk in chunks:
    if len(cur) + len(chunk) > max_chars and cur:
      groups.append(cur)
      cur = ""
    cur += chunk + "\n\n"
  if cur:
    groups.append(cur)
  return groups

def _summarize_one(text, model):
  return rag_core.llm_chat(
    [{"role": "user", "content": prompts.SUMMARY_PROMPT.format(context=text)}],
    model=model,
  )

def summarize(collection: chromadb.Collection, source: str, model: str, max_single = config.SUMMARY_SINGLE_PASS_CHARS):
  chunks = rag_core.get_doc_chunks(collection, source)
  full = "\n\n".join(chunks)
  if len(full) <= max_single:
    prompt = prompts.SUMMARY_PROMPT.format(context = full)
  else:
    groups = _group_chunks(chunks)
    partials = [_summarize_one(group, model) for group in groups]
    prompt = prompts.REDUCE_PROMPT.format(context="\n\n".join(partials))

  yield from rag_core.llm_chat(
    [{"role": "user", "content": prompt}],
    model=model,
    stream=True,
  )
