from typing import List
from collections import deque
from itertools import product
import heapq
'''
Problem Description:
Given a 2D array, we start from the top-left corner and move towards the bottom-right corner. We can move up, down, left, or right. 
The ID of a path is defined as the maximum value within that path. The goal is to find and print the path with the smallest ID among all possible paths. 
One approach that comes to mind is using DFS to traverse all paths and then selecting the result. 
During the interview, the interviewer kept asking if there is a better solution. After some thought and with time constraints, 
I ended up writing the DFS solution. I wonder if anyone has a better solution?
Suggested Solution:
For this problem, a more efficient approach than DFS is to use the Dijkstra algorithm to find a path such that the maximum value within the path is minimized.
Basic Idea:
Initialize the maximum value for all nodes to infinity.
Start from the top-left corner and use a priority queue to conduct a breadth-first search.
When considering moving from one node to another, update the value of the target node to the maximum value seen on the current path.
Once we reach the bottom-right corner, we have found a potential optimal path.
Throughout the process, we use the priority queue to always expand the node with the smallest maximum value.
This approach has a time complexity of O(mn*log(mn)), where m and n are the number of rows and columns in the 2D array respectively.
Below is the Python code using the Dijkstra algorithm for this solution: [followed by the provided code]
'''
class GraphRelatedProblems:
  def minMaxPath(self, matrix):
    if not matrix or not matrix[0]:
      return []
    rows, cols = len(matrix), len(matrix[0])
    max_values = [[float('inf')] * cols for _ in range(rows)]
    max_values[0][0] = matrix[0][0]
    pq = [(matrix[0][0], 0, 0)]  # (value, row, col)
    dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    path = {(0, 0): None}  # To reconstruct the path
    while pq:
      val, r, c = heapq.heappop(pq)
      for dr, dc in dirs:
        new_r, new_c = r + dr, c + dc
        if 0 <= new_r < rows and 0 <= new_c < cols:
          new_val = max(val, matrix[new_r][new_c])
          if new_val < max_values[new_r][new_c]:
            max_values[new_r][new_c] = new_val
            heapq.heappush(pq, (new_val, new_r, new_c))
            path[(new_r, new_c)] = (r, c)
    # Reconstruct the path
    r, c = rows - 1, cols - 1
    result_path = []
    while (r, c) is not None:
      result_path.append((r, c))
      r, c = path[(r, c)]
    return result_path[::-1]
  '''
  "Given a matrix of 0s and 1s, where 1 represents an obstacle and 0 represents a passable path, find any path from the top-left corner to the bottom-right corner and return this path. There was a minor oversight, during backtracking the 'visited' set should not be reverted along with the 'path', which can significantly optimize the time complexity."
  Here's a Python implementation using Depth-First Search:
  '''
  def find_path(self, matrix):
    if not matrix or not matrix[0]: return []
    m, n = len(matrix), len(matrix[0])
    if matrix[0][0] == 1 or matrix[m - 1][n - 1] == 1: return []
    def dfs(i, j, path, visited):
      if i == m - 1 and j == n - 1:
        return path + [(i, j)]
      visited.add((i, j))
      for x, y in [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]:
        if 0 <= x < m and 0 <= y < n and matrix[x][y] == 0 and (x, y) not in visited:
          new_path = dfs(x, y, path + [(i, j)], visited)
          if new_path:
            return new_path
      return []
    return dfs(0, 0, [], set())
  '''
  1091. Shortest Path in Binary Matrix
  Given an n x n binary matrix grid, return the length of the shortest clear path in the matrix. If there is no clear path, return -1.
A clear path in a binary matrix is a path from the top-left cell (i.e., (0, 0)) to the bottom-right cell (i.e., (n - 1, n - 1)) such that:
All the visited cells of the path are 0.
All the adjacent cells of the path are 8-directionally connected (i.e., they are different and they share an edge or a corner).
The length of a clear path is the number of visited cells of this path.
  '''
  def shortestPathBinaryMatrix_bfs(self, grid: List[List[int]]) -> int:
    INF = 10 ** 20
    R, C = len(grid), len(grid[0])
    # Check if the start or end cell is blocked
    if grid[0][0] == 1 or grid[R - 1][C - 1] == 1:
      return -1
    q = deque([(0, 0)])  # Initialize the queue with the starting cell
    dist = [[INF] * C for _ in range(R)]
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    # Initialize distance for the starting cell
    dist[0][0] = 1
    while q:
      x, y = q.popleft()
      d = dist[x][y]
      if x == R - 1 and y == C - 1:
        return d
      for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < R and 0 <= ny < C and grid[nx][ny] == 0 and dist[nx][ny] == INF:
          q.append((nx, ny))
          dist[nx][ny] = d + 1
    return -1
  '''
    A star search
    Time: O(NlogN)
    Space: O(N)
    using priority queue and introducing best_case_estimate (A*), this approach enables the performance beats 99% other python submissions
  '''
  def shortestPathBinaryMatrix_Astar(self, grid: List[List[int]]) -> int:
    max_row = len(grid) - 1
    max_col = len(grid[0]) - 1
    directions = [
      (-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    def get_neighbours(row, col):  # Helper function to find the neighbors of a given cell.
      for row_difference, col_difference in directions:
        new_row = row + row_difference
        new_col = col + col_difference
        if not (0 <= new_row <= max_row and 0 <= new_col <= max_col): continue
        if grid[new_row][new_col] != 0: continue
        yield (new_row, new_col)
    def best_case_estimate(row, col):  # Helper function for the A* heuristic.
      return max(max_row - row, max_col - col)

    if grid[0][0] or grid[max_row][max_col]: return -1  # Check that the first and last cells are open.
    visited = set()  # Set up the A* search.
    # Entries on the priority queue are of the form (total distance estimate, distance so far, (cell row, cell col))
    priority_queue = [(1 + best_case_estimate(0, 0), 1, (0, 0))]
    while priority_queue:
      estimate, distance, cell = heapq.heappop(priority_queue)
      if cell in visited: continue
      if cell == (max_row, max_col): return distance
      visited.add(cell)
      for neighbour in get_neighbours(*cell):
        if neighbour in visited: continue  # The check here isn't necessary for correctness, but it leads to a substantial performance gain.
        estimate = best_case_estimate(*neighbour) + distance + 1
        entry = (estimate, distance + 1, neighbour)
        heapq.heappush(priority_queue, entry)
    return -1  # There was no path.

'''
135. Candy
There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.
You are giving candies to these children subjected to the following requirements:
Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
Return the minimum number of candies you need to have to distribute the candies to the children.
'''
class CandyProblem:
  def candy(self, ratings: List[int]) -> int:
    '''
    [4,5,3,1,0,2,8]
    [4,5,3,1,0,2,8,9]
    [4,5,3,1,0,2,8,9,9,9]
    [4,5,3,1,0,2,8,9,9,9,10]
    [4,5]
    [1,0,2]
    '''
    dp = [1] * len(ratings)
    for i in range(1, len(ratings)):
      if ratings[i] > ratings[i - 1]:
        dp[i] = max(dp[i - 1] + 1, dp[i])
    for i in range(len(ratings) - 2, -1, -1):
      if ratings[i] > ratings[i + 1]:
        dp[i] = max(dp[i + 1] + 1, dp[i])
    return sum(dp)

'''
200. Number of Islands
'''
class NumberIslands:
  def numIslands_rec(self, grid: List[List[str]]) -> int:
    m, n = len(grid), len(grid[0])
    directs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    def dfs(i, j):
      grid[i][j] = 'T'
      for direct in directs:
        x, y = i + direct[0], j + direct[1]
        if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
          dfs(x, y)
    count = 0
    for i, j in product(range(m), range(n)):
      if grid[i][j] == '1':
        dfs(i, j)
        count += 1
    return count

  def numIslands_iter(self, grid: List[List[str]]) -> int:
    if not grid: return 0
    m, n = len(grid), len(grid[0])
    directs = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    count = 0
    for i, j in product(range(m), range(n)):
      if grid[i][j] == '1':
        # Start BFS from here and increment the island count
        queue = deque([(i, j)])
        grid[i][j] = 'T'
        while queue:
          i1, j1 = queue.popleft()
          for direct in directs:
            x, y = i1 + direct[0], j1 + direct[1]
            if 0 <= x < m and 0 <= y < n and grid[x][y] == '1':
              queue.append((x, y))  # Change from (i1, j1) to (x, y)
              grid[x][y] = 'T'  # Mark the visited position with 'T'
        count += 1  # Increment count inside the if condition, not outside
    return count


'''
994. Rotting Oranges
You are given an m x n grid where each cell can have one of three values:

0 representing an empty cell,
1 representing a fresh orange, or
2 representing a rotten orange.
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.
Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return -1.
Input: grid = [[2,1,1],[1,1,0],[0,1,1]]
Output: 4
Input: grid = [[2,1,1],[0,1,1],[1,0,1]]
Output: -1
Explanation: The orange in the bottom left corner (row 2, column 0) is never rotten, because rotting only happens 4-directionally.
Example 3:

Input: grid = [[0,2]]
Output: 0
Explanation: Since there are already no fresh oranges at minute 0, the answer is just 0.
'''
class RottingOranges:
  def orangesRotting_bfs(self, grid: List[List[int]]) -> int:
    queue = deque()
    fresh_oranges = 0 # Step 1). build the initial set of rotten oranges
    rows, cols = len(grid), len(grid[0])
    for r in range(rows):
      for c in range(cols):
        if grid[r][c]==2:
          queue.append((r, c))
        elif grid[r][c]==1:
          fresh_oranges += 1
    queue.append(((-1, -1))) # Mark the round / level, _i.e_ the ticker of timestamp
    minutes_elapsed = -1 # Step 2). start the rotting process via BFS
    directions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
    while queue:
      row, col = queue.popleft()
      if row == -1:  # We finish one round of processing
        minutes_elapsed += 1
        if queue: queue.append((-1, -1))  # to avoid the endless loop
      else:  # this is a rotten orange # then it would contaminate its neighbors
        for d in directions:
          neighbor_row, neighbor_col = row + d[0], col + d[1]
          if rows > neighbor_row >= 0 and cols > neighbor_col >= 0:
            if grid[neighbor_row][neighbor_col] == 1:
              grid[neighbor_row][neighbor_col] = 2  # this orange would be contaminated
              fresh_oranges -= 1
              queue.append((neighbor_row, neighbor_col))  # this orange would then contaminate other oranges
    return minutes_elapsed if fresh_oranges == 0 else -1  # return elapsed minutes if no fresh orange left

  '''
  modify from this https://leetcode.com/problems/rotting-oranges/solutions/3600345/dfs-solution-explained-in-figures-python/
  '''
  def dfs(self, rotten_after, grid, row, col):
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    for dr, dc in directions:
      newRow, newCol = row + dr, col + dc
      if 0 <= newRow < len(grid) and 0 <= newCol < len(grid[0]) and grid[newRow][newCol] == 1:
        if rotten_after[newRow][newCol] == 0 or rotten_after[newRow][newCol] > rotten_after[row][col] + 1:
          rotten_after[newRow][newCol] = rotten_after[row][col] + 1
          self.dfs(rotten_after, grid, newRow, newCol)

  def orangesRotting(self, grid: List[List[int]]) -> int:
    # Get rows and columns numbers
    rows, cols = len(grid), len(grid[0])
    if rows == 0: return -1

    # Define rotten_after
    rotten_after = [[0] * cols for _ in range(rows)]

    # Execute DFS for all rotten oranges
    for row in range(rows):
      for col in range(cols):
        if grid[row][col] == 2:
          self.dfs(rotten_after, grid, row, col)

    # Check if there are any fresh oranges left
    for row in range(rows):
      for col in range(cols):
        if grid[row][col] == 1 and rotten_after[row][col] == 0:
          return -1

    # Return the maximum time from the rotten_after array
    return max(map(max, rotten_after))