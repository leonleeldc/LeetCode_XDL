'''
317. Shortest Distance from All Buildings
https://leetcode.com/problems/shortest-distance-from-all-buildings/description/
You are given an m x n grid grid of values 0, 1, or 2, where:
each 0 marks an empty land that you can pass by freely,
each 1 marks a building that you cannot pass through, and
each 2 marks an obstacle that you cannot pass through.
You want to build a house on an empty land that reaches all buildings in the shortest total travel distance. You can only move up, down, left, and right.
Return the shortest travel distance for such a house. If it is not possible to build such a house according to the above rules, return -1.
The total travel distance is the sum of the distances between the houses of the friends and the meeting point.
The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.
'''
from typing import List
from collections import deque
class Solution:
  def shortestDistance_bfs_v1(self, grid: List[List[int]]) -> int:
    n = len(grid);
    m = len(grid[0])
    res = [[0] * m for _ in range(n)]
    mark = [[0] * m for _ in range(n)]
    def bfs(si, sj):
      visited = set()
      q = deque([(si, sj)])
      visited.add((si, sj))
      dirs = [[0, 1], [1, 0], [0, -1], [-1, 0]]
      n = len(grid);
      m = len(grid[0])
      dis = 0
      while q:
        length = len(q)
        for _ in range(length):
          x, y = q.popleft()
          if grid[x][y] == 0:
            res[x][y] += dis
            mark[x][y] += 1
          for dx, dy in dirs:
            xx = x + dx
            yy = y + dy
            if (xx, yy) not in visited and xx >= 0 and xx < n and yy >= 0 and yy < m and grid[xx][yy] == 0:
              q.append((xx, yy))
              visited.add((xx, yy))
        dis += 1
    min_dis = float('inf')
    cnt = 0
    for i in range(len(grid)):
      for j in range(len(grid[0])):
        if grid[i][j] == 1:
          cnt += 1
          bfs(i, j)
    for i in range(len(grid)):
      for j in range(len(grid[0])):
        if mark[i][j] == cnt:
          if min_dis > res[i][j]:
            min_dis = res[i][j]
    if min_dis == float('inf'):
      return -1
    else:
      return min_dis

  def shortestDistance_bfs_v2(self, grid):
    if not grid or not grid[0]: return -1
    M, N, buildings = len(grid), len(grid[0]), sum(val for line in grid for val in line if val == 1)
    hit, distSum = [[0] * N for i in range(M)], [[0] * N for i in range(M)]
    def BFS(start_x, start_y):
      visited = [[False] * N for k in range(M)]
      visited[start_x][start_y], count1, queue = True, 1, deque([(start_x, start_y, 0)])
      while queue:
        x, y, dist = queue.popleft()
        for i, j in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
          if 0 <= i < M and 0 <= j < N and not visited[i][j]:
            visited[i][j] = True
            if not grid[i][j]:
              queue.append((i, j, dist + 1))
              hit[i][j] += 1
              distSum[i][j] += dist + 1
            elif grid[i][j] == 1:
              count1 += 1
      return count1 == buildings

    for x in range(M):
      for y in range(N):
        if grid[x][y] == 1:
          if not BFS(x, y): return -1
    return min([distSum[i][j] for i in range(M) for j in range(N) if not grid[i][j] and hit[i][j] == buildings] or [-1])