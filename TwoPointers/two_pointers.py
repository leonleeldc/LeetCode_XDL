'''
11. Container With Most Water
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the ith line are (i, 0) and (i, height[i]).

Find two lines that together with the x-axis form a container, such that the container contains the most water.

Return the maximum amount of water a container can store.
'''
from typing import List
class TwoPointersRelated:
  def maxAreaLR(self, height: List[int]) -> int:
    max_area, left, right = 0, 0, len(height) - 1
    while left < right:
      width = right - left
      if height[left] < height[right]:
        max_area = max(max_area, width * height[left])
        left += 1
      else:
        max_area = max(max_area, width * height[right])
        right -= 1
    return max_area

  def maxAreaP1P2(self, height: List[int]) -> int:
    if len(height) <= 1: return 0
    p1, p2 = 0, len(height) - 1
    max_area = float('-inf')
    while p1 < p2:
      area = height[p1] * (p2 - p1) if height[p1] < height[p2] else height[p2] * (p2 - p1)
      if max_area < area:
        max_area = area
      if height[p1] < height[p2]:
        p1 += 1
      else:
        p2 -= 1
    return max_area