'''
269. Alien Dictionary https://leetcode.com/problems/alien-dictionary/description/
'''
from typing import List
class TopologicalSorting:
  def alienOrder(self, words: List[str]) -> str:
    adj = {c: set() for word in words for c in word}
    for i in range(1, len(words)):
      word, word_next = words[i - 1], words[i]
      min_len = min(len(word), len(word_next))
      if len(word) > len(word_next) and word[:min_len] == word_next[:min_len]:
        return ''
      for j in range(min_len):
        if word[j] != word_next[j]:
          adj[word[j]].add(word_next[j])
          break
    res, word_leave = [], {}

    def dfs(c):
      if c in word_leave: return word_leave[c]
      word_leave[c] = True
      for neigh in adj[c]:
        if dfs(neigh):
          return True  # there is a cycle in this case
      res.append(c)
      word_leave[c] = False

    for c in adj:
      if dfs(c):
        return ''
    res.reverse()
    return ''.join(res)

