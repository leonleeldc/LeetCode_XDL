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
