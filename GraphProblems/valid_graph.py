from typing import List
from collections import deque
class UnionFind:
  def __init__(self, n):
    self.parent = [node for node in range(n)]

  def find(self, A):
    while A != self.parent[A]:
      A = self.parent[A]
    return A

  def union(self, A, B):
    rootA = self.find(A)
    rootB = self.find(B)
    if rootA == rootB: return False
    self.parent[rootB] = rootA
    return True

'''
261 Graph Valid Tree
You have a graph of n nodes labeled from 0 to n - 1. You are given an integer n and a list of edges where edges[i] = [ai, bi] indicates that there is an undirected edge between nodes ai and bi in the graph.
Return true if the edges of the given graph make up a valid tree, and false otherwise.
Input: n = 5, edges = [[0,1],[0,2],[0,3],[1,4]]
Output: true

Input: n = 5, edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
Output: false
'''
class GraphValidTree:
  def validTree_uf(self, n: int, edges: List[List[int]]) -> bool:
    uf = UnionFind(n)
    for edge in edges:
      if not uf.union(edge[0], edge[1]):
        return False
    return True

  def validTree_bfs(self, n: int, edges: List[List[int]]) -> bool:
    ##Approach 1: Graph Theory + Iterative Breadth-First Search
    if len(edges) != n - 1: return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
      adj[a].append(b)
      adj[b].append(a)
    queue, seen = deque([0]), {0}
    while queue:
      node = queue.popleft()
      neibors = adj[node]
      for neibor in neibors:
        if neibor not in seen:
          seen.add(neibor)
          queue.append(neibor)
    return len(seen) == n

  def validTree_dfs(self, n: int, edges: List[List[int]]) -> bool:
    ##Approach 1: Graph Theory + Iterative Breadth-First Search
    if len(edges) != n - 1: return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
      adj[a].append(b)
      adj[b].append(a)
    seen = set()
    def dfs(node):
      if node in seen: return
      seen.add(node)
      for neibor in adj[node]:
        dfs(neibor)
    dfs(0)
    return len(seen) == n

  def validTree_dfs_iter(self, n: int, edges: List[List[int]]) -> bool:
    # if len(edges)!=n-1: return False
    adj = [[] for _ in range(n)]
    for a, b in edges:
      adj[a].append(b)
      adj[b].append(a)
    stack, parent = [0], {0: -1}
    while stack:
      node = stack.pop()
      for neigh in adj[node]:
        if neigh == parent[node]:
          continue
        if neigh in parent:
          ##usually, false will stop here if the tree is not valid. Namely, loops are found here.
          return False
        parent[neigh] = node
        stack.append(neigh)
    # print(f'len(parent)={len(parent)} and n={n}')
    return len(parent) == n