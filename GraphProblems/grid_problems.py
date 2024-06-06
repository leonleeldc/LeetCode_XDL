from typing import List
from collections import deque
'''
1293. Shortest Path in a Grid with Obstacles Elimination
You are given an m x n integer matrix grid where each cell is either 0 (empty) or 1 (obstacle). You can move up, down, left, or right from and to an empty cell in one step.
Return the minimum number of steps to walk from the upper left corner (0, 0) to the lower right corner (m - 1, n - 1) given that you can eliminate at most k obstacles. If it is not possible to find such walk return -1.
Example 1
Input: grid = [[0,0,0],[1,1,0],[0,0,0],[0,1,1],[0,0,0]], k = 1
Output: 6
Explanation:
The shortest path without eliminating any obstacle is 10.
The shortest path with one obstacle elimination at position (3,2) is 6. Such path is (0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (3,2) -> (4,2).
Example 2:
Input: grid = [[0,1,1],[1,1,1],[1,0,0]], k = 1
Output: -1
Explanation: We need to eliminate at least two obstacles to find such a walk.
'''
class ShortestPathGridObstacleElim:
  def shortestPath(self, grid: List[List[int]], k: int) -> int:
    rows, cols = len(grid), len(grid[0])
    target = (rows - 1, cols - 1)
    if k >= rows + cols - 2: return rows + cols - 2
    queue = deque([(0, (0, 0, k))])
    seen = set([(0, 0, k)])
    while queue:
      steps, (r, c, k) = queue.popleft()
      if (r, c) == target: return steps
      for i, j in ([-1, 0], [1, 0], [0, -1], [0, 1]):
        x, y = r + i, c + j
        if 0 <= x < rows and 0 <= y < cols:
          rem = k - grid[x][y]
          if rem >= 0 and (x, y, rem) not in seen:
            seen.add((x, y, rem))
            queue.append((steps + 1, (x, y, rem)))
    return -1

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
  '''
  490. The Maze
  There is a ball in a maze with empty spaces (represented as 0) and walls (represented as 1). The ball can go through the empty spaces by rolling up, down, left or right, but it won't stop rolling until hitting a wall. When the ball stops, it could choose the next direction.
  Given the m x n maze, the ball's start position and the destination, where start = [startrow, startcol] and destination = [destinationrow, destinationcol], return true if the ball can stop at the destination, otherwise return false.
  You may assume that the borders of the maze are all walls (see examples).
  '''
  def hasPath_dfs(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
    m, n = len(maze), len(maze[0])
    visited = [[False] * n for _ in range(m)]
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    def dfs(x, y):
      if [x, y] == destination: return True
      if visited[x][y]: return False
      visited[x][y] = True
      for dx, dy in directions:
        xx, yy = x, y
        while 0 <= xx + dx < m and 0 <= yy + dy < n and maze[xx + dx][yy + dy] == 0:
          xx += dx
          yy += dy
        if dfs(xx, yy): return True
      return False
    return dfs(start[0], start[1])
  def hasPath_bfs(self, maze: List[List[int]], start: List[int], destination: List[int]) -> bool:
    queue = deque([start])
    # There is a trick not to use a set.
    # You can find it in
    # https://leetcode.com/problems/the-maze/discuss/97074/Python-BFS-solution
    visited = set()
    m, n = len(maze), len(maze[0])
    while queue:
      x, y = queue.popleft()
      visited.add((x, y))
      if [x, y] == destination: return True
      x0, y0 = x, y
      for delta_x, delta_y in [[0, 1], [0, -1], [1, 0], [-1, 0]]:
        while 0 <= x + delta_x < m and 0 <= y + delta_y < n and not maze[x + delta_x][y + delta_y]:
          x += delta_x
          y += delta_y
        else:
          if not (x, y) in visited: queue.append([x, y])
        x = x0
        y = y0
    return False

