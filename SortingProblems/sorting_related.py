'''
164. Maximum Gap
Given an integer array nums, return the maximum difference between two successive elements in its sorted form. If the array contains less than two elements, return 0.
You must write an algorithm that runs in linear time and uses linear extra space.
Example 1:
Input: nums = [3,6,9,1]
Output: 3
Explanation: The sorted form of the array is [1,3,6,9], either (3,6) or (6,9) has the maximum difference 3.
Example 2:

Input: nums = [10]
Output: 0
Explanation: The array contains less than 2 elements, therefore return 0.
'''
from typing import List
import heapq
'''
315. Count of Smaller Numbers After Self
Given an integer array nums, return an integer array counts where counts[i] is the number of smaller elements to the right of nums[i].
Example 1:

Input: nums = [5,2,6,1]
Output: [2,1,1,0]
Explanation:
To the right of 5 there are 2 smaller elements (2 and 1).
To the right of 2 there is only 1 smaller element (1).
To the right of 6 there is 1 smaller element (1).
To the right of 1 there is 0 smaller element.
Example 2:

Input: nums = [-1]
Output: [0]
Example 3:

Input: nums = [-1,-1]
Output: [0,0]

'''


class TreeNode:
  def __init__(self, value):
    self.value = value
    self.num_left = 0
    self.count = 0
    self.left = None
    self.right = None


class CountSmallerNumbersAfterSelf:
  def countSmallerMergeSort(self, nums: List[int]) -> List[int]:
    output = [0] * len(nums)
    indexed_nums = [(num, i) for i, num in enumerate(nums)]
    def merge_sort_like(arr):
      if len(arr) > 1:
        # Find the middle point and divide the array into two halves
        mid = len(arr) // 2
        left_half = arr[:mid]
        right_half = arr[mid:]
        # Recursively sort the two halves
        merge_sort_like(left_half)
        merge_sort_like(right_half)
        # Merge the sorted halves
        i = j = k = 0
        right_count = 0
        while i < len(left_half) and j < len(right_half):
          if left_half[i][0] <= right_half[j][0]:
            arr[k] = left_half[i]
            output[left_half[i][1]] += right_count
            i += 1
          else:
            arr[k] = right_half[j]
            right_count += 1
            j += 1
          k += 1
        while i < len(left_half):
          arr[k] = left_half[i]
          output[left_half[i][1]] += right_count
          i += 1
          k += 1
        while j < len(right_half):
          arr[k] = right_half[j]
          j += 1
          k += 1
    merge_sort_like(indexed_nums)
    return output
  def countSmaller_bt(self, nums: List[int]) -> List[int]:
    if not nums:
      return []
    nodes = sorted(set(nums))
    root = self.construct(nodes, 0, len(nodes) - 1)
    res = []
    for node in reversed(nums):
      res.append(self.insert(root, node))
    return res[::-1]
  def insert(self, root, node):
    res = 0
    while root.value != node:
      if node < root.value:
        root.num_left += 1
        if not root.left:
          root.left = TreeNode(node)
        root = root.left
      else:
        res += root.count + root.num_left
        if not root.right:
          root.right = TreeNode(node)
        root = root.right
    root.count += 1
    return res + root.num_left

  def construct(self, nodes, left, right):
    if left > right:
      return None
    mid = left + (right - left) // 2
    root = TreeNode(nodes[mid])
    root.left = self.construct(nodes, left, mid - 1)
    root.right = self.construct(nodes, mid + 1, right)
    return root

class MaxGap:
  def maximumGap(self, nums: List[int]) -> int:
    if len(nums) < 2: return 0
    max_num, min_num = max(nums), min(nums)
    # Case when all numbers are the same
    if max_num == min_num: return 0
    # Initialize buckets
    size = (max_num - min_num) // (len(nums) - 1) or 1
    buckets = [[float('inf'), float('-inf')] for _ in range((max_num - min_num) // size + 1)]
    # Populate the buckets
    for num in nums:
      b = buckets[(num - min_num) // size]
      b[0] = min(b[0], num)
      b[1] = max(b[1], num)
    # Compute the max gap
    max_gap = 0
    prev_max = min_num
    for b in buckets:
      if b[0] == float('inf'):  # Skip empty buckets
        continue
      max_gap = max(max_gap, b[0] - prev_max)
      prev_max = b[1]
    return max_gap
  def maximumGap_heap(self, nums: List[int]) -> int:
      if len(nums)<2: return 0
      heap = []
      heapq.heapify(heap)
      for i, num in enumerate(nums):
          heapq.heappush(heap, (num, i))
      max_gap = 0
      first = heapq.heappop(heap)[0]
      while heap:
          second = heapq.heappop(heap)[0]
          max_gap = max(max_gap, second-first)
          first = second
      return max_gap
'''
462. Minimum Moves to Equal Array Elements II
'''
class MinMoves:
  def minMoves2(self, nums: List[int]) -> int:
    n = len(nums)
    mid = n // 2
    nums.sort()
    res = 0
    for i in range(n):
      res = res + abs(nums[i] - nums[mid])
    return res

  '''
  296. Best Meeting Point
  Given an m x n binary grid grid where each 1 marks the home of one friend, return the minimal total travel distance.
The total travel distance is the sum of the distances between the houses of the friends and the meeting point.
The distance is calculated using Manhattan Distance, where distance(p1, p2) = |p2.x - p1.x| + |p2.y - p1.y|.
  '''
  def minTotalDistance(self, grid):
    """
    :type grid: List[List[int]]
    :rtype: int
    """
    y_coords = []
    x_coords = []
    for y in range(len(grid)):
      for x in range(len(grid[0])):
        if grid[y][x] == 1:
          y_coords.append(y)
          x_coords.append(x)
    y_coords = sorted(y_coords)
    x_coords = sorted(x_coords)
    n = len(y_coords)
    meet_point = (y_coords[n // 2], x_coords[n // 2])
    res = 0
    for i in range(n):
      res += abs(meet_point[0] - y_coords[i]) + abs(meet_point[1] - x_coords[i])
    return res
  def minTotalDistance_v2(self, grid):
    total = 0
    for grid in grid, zip(*grid):
      X = []
      for x, row in enumerate(grid):
        X += [x] * sum(row)
      total += sum(abs(x - X[len(X) // 2])
                   for x in X)
    return total
  def minTotalDistance_v3(self, grid):
    row_sum = list(map(sum, grid))
    col_sum = list(map(sum, zip(*grid)))  # syntax sugar learned from stefan :-)
    def minTotalDistance1D(vec):
      i, j = -1, len(vec)
      d = left = right = 0
      while i != j:
        if left < right:
          d += left
          i += 1
          left += vec[i]
        else:
          d += right
          j -= 1
          right += vec[j]
      return d
    return minTotalDistance1D(row_sum) + minTotalDistance1D(col_sum)
