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
'''
2. Trapping Rain Water
Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it can trap after raining.
'''
class TrappingRainWater:
  def trap_lr_max_twodir(self, height: List[int]) -> int:
    left_max, right_max = [0] * len(height), [0] * len(height)
    for i in range(1, len(height)):
      left_max[i] = max(left_max[i - 1], height[i - 1])
    for i in range(len(height) - 2, -1, -1):
      right_max[i] = max(right_max[i + 1], height[i + 1])
    res = 0
    for i in range(len(height)):
      res += max(0, min(left_max[i], right_max[i]) - height[i])
    return res
  def trap_monostack(self, height: List[int]) -> int:
      ans, cur, n, stack=0, 0, len(height), []
      while cur<n:
          while stack and height[cur]>height[stack[-1]]:
              top = stack.pop()
              if not stack: break
              dist = cur-stack[-1]-1
              bounded_height=min(height[cur], height[stack[-1]])-height[top]
              ans += dist*bounded_height
          stack.append(cur)
          cur+=1
      return ans
  def trap_lr_samedir(self, height: List[int]) -> int:
      if len(height)==0: return 0
      left, right = 0, len(height)-1
      max_l, max_r, ans = height[left], height[right], 0
      while left<right:
          if max_l<max_r:
              ans += max_l - height[left]
              left+=1
              max_l = max(max_l, height[left])
          else:
              ans += max_r - height[right]
              right-=1
              max_r = max(max_r, height[right])
      return ans

  def trap_lr_max_twodir_optimized(self, height: List[int]) -> int:
    if len(height) == 0: return 0
    left, right = 0, len(height) - 1
    max_l, max_r, ans = height[left], height[right], 0
    while left < right:
      if max_l < max_r:
        ans += max_l - height[left]
        left += 1
        max_l = max(max_l, height[left])
      else:
        ans += max_r - height[right]
        right -= 1
        max_r = max(max_r, height[right])
    return ans
  def trap_rec(self, height: List[int]) -> int:
      if height==[] or len(height)<2:
          return 0
      stored=0
      min_height=1
      mark=-1
      for j,i in enumerate(height):
          if i>=min_height:
              min_height=i
              if mark>-1:
                  stored+=height[mark]*(j-mark)-sum([height[x] for x in range(mark,j)])
              mark=j
      if min_height>height[-1]:
          stored+=self.trap_rec(list(reversed(height[mark:])))
      return stored

