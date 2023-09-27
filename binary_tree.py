'''
993. Cousins in Binary Tree
Given the root of a binary tree with unique values and the values of two different nodes of the tree x and y, return true if the nodes corresponding to the values x and y in the tree are cousins, or false otherwise.
Two nodes of a binary tree are cousins if they have the same depth with different parents.
Note that in a binary tree, the root node is at the depth 0, and children of each depth k node are at the depth k + 1.
Input: root = [1,2,3,4], x = 4, y = 3
Output: false
Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
Output: true
Constraints:
The number of nodes in the tree is in the range [2, 100].
1 <= Node.val <= 100
Each node has a unique value.
x != y
x and y are exist in the tree.
'''
# Definition for a binary tree node.
from collections import deque
from typing import Optional, List
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
class CousinsBinaryTree:
  def __init__(self):
    # To save the depth of the first node.
    self.recorded_depth = None
    self.is_cousin = False
  def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
    # Queue for BFS
    queue = deque([root])
    while queue:
      siblings = False
      cousins = False
      nodes_at_depth = len(queue)
      for _ in range(nodes_at_depth):
        # FIFO
        node = queue.popleft()
        # Encountered the marker.
        # Siblings should be set to false as we are crossing the boundary.
        if node is None:
          siblings = False
        else:
          if node.val == x or node.val == y:
            # Set both the siblings and cousins flag to true
            # for a potential first sibling/cousin found.
            if not cousins:
              siblings, cousins = True, True
            else:
              # If the siblings flag is still true this means we are still
              # within the siblings boundary and hence the nodes are not cousins.
              return not siblings
          queue.append(node.left) if node.left else None
          queue.append(node.right) if node.right else None
          # Adding the null marker for the siblings
          queue.append(None)
      # After the end of a level if `cousins` is set to true
      # This means we found only one node at this level
      if cousins: return False
    return False

  def dfs(self, node, depth, x, y):
    if node is None: return False
    # Don't go beyond the depth restricted by the first node found.
    if self.recorded_depth and depth > self.recorded_depth: return False
    if node.val == x or node.val == y:
      if self.recorded_depth is None:
        # Save depth for the first node.
        self.recorded_depth = depth
      # Return true, if the second node is found at the same depth.
      return self.recorded_depth == depth
    left = self.dfs(node.left, depth + 1, x, y)
    right = self.dfs(node.right, depth + 1, x, y)
    # self.recorded_depth != depth + 1 would ensure node x and y are not
    # immediate child nodes, otherwise they would become siblings.
    if left and right and self.recorded_depth != depth + 1:
      self.is_cousin = True
    return left or right

  def isCousins_dfs(self, root: TreeNode, x: int, y: int) -> bool:
    # Recurse the tree to find x and y
    self.dfs(root, 0, x, y)
    return self.is_cousin

  def isCousins_bfs(self, root: Optional[TreeNode], x: int, y: int) -> bool:
    queue = deque([(root, 0)])
    x_level, x_parent, y_level, y_parent = 0, None, 0, None
    while queue:
      size = len(queue)
      for _ in range(size):
        node, level = queue.popleft()
        if node.left:
          if node.left.val == x:
            x_level = level + 1
            x_parent = node.val
          if node.left.val == y:
            y_level = level + 1
            y_parent = node.val
          queue.append((node.left, level + 1))
        if node.right:
          if node.right.val == x:
            x_level = level + 1
            x_parent = node.val
          if node.right.val == y:
            y_level = level + 1
            y_parent = node.val
          queue.append((node.right, level + 1))
    return x_parent != y_parent and x_level == y_level

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
  modify from this https://leetcode.com/problems/rotting-oranges/
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


