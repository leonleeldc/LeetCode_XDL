'''
678. Valid Parenthesis String
Given a string s containing only three types of characters: '(', ')' and '*', return true if s is valid.

The following rules define a valid string:

Any left parenthesis '(' must have a corresponding right parenthesis ')'.
Any right parenthesis ')' must have a corresponding left parenthesis '('.
Left parenthesis '(' must go before the corresponding right parenthesis ')'.
'*' could be treated as a single right parenthesis ')' or a single left parenthesis '(' or an empty string "".
'''
from collections import deque
from typing import List


class RLEIterator:

  def __init__(self, encoding: List[int]):
    self.encoding = encoding
    self.idx = 0

  def next(self, n: int) -> int:
    while n > 0 and self.idx < len(self.encoding):
      if n < self.encoding[self.idx]:
        self.encoding[self.idx] -= n
        return self.encoding[self.idx + 1]
      else:
        answer = self.encoding[self.idx + 1]
        n -= self.encoding[self.idx]
        self.idx += 2
        if n == 0:
          return answer
    return -1 if self.idx >= len(self.encoding) else answer
class RunLengthIterator:
  def __init__(self, encoded):
    self.encoded = encoded
    self.index = 0
    self.remaining = 0
    self.value = None

  def __iter__(self):
    return self

  def __next__(self):
    if self.remaining == 0:
      if self.index >= len(self.encoded):
        raise StopIteration
      self.remaining = self.encoded[self.index]
      self.value = self.encoded[self.index + 1]
      self.index += 2

    self.remaining -= 1
    return self.value

class ChunkReceiver:
  def __init__(self):
    self.received_chunks = {}
    self.expected_chunk = 0

  def add_chunk(self, chunk_id, data):
    self.received_chunks[chunk_id] = data
    self._update_expected_chunk()

  def _update_expected_chunk(self):
    while self.expected_chunk in self.received_chunks:
      self.expected_chunk += 1

  def get_expected_chunk(self):
    return self.expected_chunk




from functools import lru_cache
class ValidParenthesisString:
  def checkValidStringIter(self, s: str) -> bool:
    min_op, max_op = 0, 0
    for c in s:
      min_op +=1 if c=='(' else -1
      max_op += 1 if c!=')' else -1
      if max_op < 0: break
      min_op = max(min_op, 0)
    return min_op==0

  def checkValidString_queue(self, s: str) -> bool:
    q = deque([(0, "")]) # Initialize the queue with initial state
    for c in s:
      next_states = set()       # Store the new states after processing current character
      while q:
        i, track = q.popleft()
        if c == '(':
          next_states.add((i + 1, track + '('))
        elif c == ')':
          if track and track[-1] == '(':
            next_states.add((i + 1, track[:-1]))
        elif c == '*':
          next_states.add((i + 1, track + '(')) # Consider * as (
          if track and track[-1] == '(': # Consider * as )
            next_states.add((i + 1, track[:-1]))
          next_states.add((i + 1, track)) # Consider * as nothing
      q.extend(next_states) # Add the new states to queue
    for _, track in q: # Check if one of the final states is valid
      if not track: return True
    return False

  def checkValidString_dfs(self, s: str) -> bool:
    @lru_cache(None)
    def dfs(track, s):
      if not s: return not track
      for c in s:
        if c == '(':
          return dfs(track + '(', s[1:])
        elif c == ')':
          if not track or track[-1] != '(': return False
          return dfs(track[:-1], s[1:])
        elif c == '*':
          return dfs(track, '(' + s[1:]) or dfs(track, ')' + s[1:]) or dfs(track, s[1:])

    return dfs('', s)