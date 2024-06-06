import math
from collections import deque
class MRUQueue:
  def __init__(self, n: int):
    self.chunk_size = int(math.sqrt(n))
    self.chunks = []
    self.n = n
    current_chunk = deque()
    for i in range(1, n + 1):
      current_chunk.append(i)
      if len(current_chunk) == self.chunk_size:
        self.chunks.append(current_chunk)
        current_chunk = deque()
    if current_chunk:
      self.chunks.append(current_chunk)

  def fetch(self, k: int) -> int:
    # Find the chunk and the position within the chunk
    chunk_index = (k - 1) // self.chunk_size
    position_within_chunk = (k - 1) % self.chunk_size

    # Extract the element
    target_chunk = self.chunks[chunk_index]
    value = target_chunk[position_within_chunk]

    # Remove the element from the chunk
    del target_chunk[position_within_chunk]

    # Ensure the chunk stays the correct size
    for i in range(chunk_index, len(self.chunks) - 1):
      if len(self.chunks[i]) < self.chunk_size:
        self.chunks[i].append(self.chunks[i + 1].popleft())

    # Add the element to the last chunk
    if len(self.chunks[-1]) < self.chunk_size:
      self.chunks[-1].append(value)
    else:
      self.chunks.append(deque([value]))
    return value
