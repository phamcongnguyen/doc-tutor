from __future__ import annotations

from collections.abc import Callable

import ollama

# Callback báo tiến độ embed: nhận (số chunk đã xong, tổng số chunk).
# Kiểu trả về là object vì giá trị trả về bị bỏ qua — lambda trả gì cũng được.
ProgressCallback = Callable[[int, int], object]


class OllamaEmbedder:
  """Chuyển text thành vector embedding qua Ollama, theo từng batch nhỏ.

  Batch giúp file lớn không bị dồn vào một request khổng lồ (dễ timeout,
  lỗi giữa chừng mất hết) và cho phép báo tiến độ qua on_progress(done, total).
  """

  def __init__(self, model: str, batch_size: int) -> None:
    self.model = model
    self.batch_size = batch_size

  def embed(self, texts: list[str],
            on_progress: ProgressCallback | None = None) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), self.batch_size):
      out = ollama.embed(model=self.model, input=texts[i:i + self.batch_size])
      vectors.extend(out["embeddings"])
      if on_progress:
        on_progress(min(i + self.batch_size, len(texts)), len(texts))
    return vectors
