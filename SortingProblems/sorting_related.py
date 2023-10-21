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
