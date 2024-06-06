'''
207. Course Schedule
There are a total of numCourses courses you have to take,
labeled from 0 to numCourses - 1. You are given an array
prerequisites where prerequisites[i] = [ai, bi]
indicates that you must take course bi first if you want to take course ai.

For example, the pair [0, 1], indicates that to take course 0 you have to first take course 1.
Return true if you can finish all courses. Otherwise, return false.
'''
from typing import List
class CourseSchedule:
  def canFinish(self, numCourses: int, prerequisites: List[List[int]]) -> bool:
    g = [0] * numCourses
    e = [[] for _ in range(numCourses)]
    for u, v in prerequisites:
      g[u] += 1
      e[v].append(u)

    stack = [idx for idx, u in enumerate(g) if u == 0]

    while stack:
      idx = stack.pop()
      e_u = e[idx]
      for u in e_u:
        g[u] -= 1
        if g[u] == 0:
          stack.append(u)
    return not any(g)

'''
269. Alien Dictionary https://leetcode.com/problems/alien-dictionary/description/
Example 1:

Input: words = ["wrt","wrf","er","ett","rftt"]
Output: "wertf"
Example 2:

Input: words = ["z","x"]
Output: "zx"
Example 3:

Input: words = ["z","x","z"]
Output: ""
Explanation: The order is invalid, so return "".
'''
from typing import List
from collections import deque, defaultdict, Counter
class TopologicalSorting:
  def alienOrder_dfs(self, words: List[str]) -> str:
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
      if c in word_leave:
        return word_leave[c]
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

  def alienOrder_bfs(self, words: List[str]) -> str:
    # Step 1: Create a graph
    adj_list = defaultdict(set)
    # Count indegree for each character
    in_degree = Counter({c: 0 for word in words for c in word})
    for i in range(1, len(words)):
      w1, w2 = words[i - 1], words[i]
      for c, d in zip(w1, w2):
        if c != d:
          if d not in adj_list[c]:
            adj_list[c].add(d)
            in_degree[d] += 1
          break
      else:
        if len(w1) > len(w2):
          return ''
    # Step 2: BFS
    queue = deque([c for c, count in in_degree.items() if count == 0])
    output = []
    while queue:
      c = queue.popleft()
      output.append(c)
      for d in adj_list[c]:
        in_degree[d] -= 1
        if in_degree[d] == 0:
          queue.append(d)
    # if not all characters are in output, it means there's a cycle.
    if len(output) < len(in_degree):
      return ''
    return ''.join(output)

