'''
74. Search a 2D Matrix
Solved
Medium
Topics
Companies
You are given an m x n integer matrix matrix with the following two properties:

Each row is sorted in non-decreasing order.
The first integer of each row is greater than the last integer of the previous row.
Given an integer target, return true if target is in matrix or false otherwise.

You must write a solution in O(log(m * n)) time complexity.
'''
from typing import List
class Searc2DhMat:
  def searchMatrix(self, matrix: List[List[int]], target: int) -> bool:
    up, down = 0, len(matrix) - 1
    while up < down:
      mid = (up + down) // 2
      if matrix[mid][0] < target:
        up = mid + 1
      else:
        down = mid - 1
    l, r = 0, len(matrix[0]) - 1
    while l <= r:
      mid = (l + r) // 2
      if matrix[up][mid] < target:
        l = mid + 1
      else:
        r = mid - 1
    print(f'up=={up} and l={l}')
    print(f'matrix[up][l]={matrix[up][l]}')
    return True if matrix[up][l] == target else False

'''
https://www.youtube.com/watch?v=J-IQxfYRTto, good explanation given by huahua leetcode
278. First Bad Version
'''
class FirstBadVersion:
  def isBadVersion(self, num):
    pass
  def firstBadVersion_rec(self, n: int) -> int:
    def binary_search(l, h):
      m = (l + h) // 2
      if self.isBadVersion(m):
        return binary_search(l, m - 1)
      elif self.isBadVersion(m + 1):
        return m + 1
      else:
        return binary_search(m + 1, h)
    return binary_search(0, n - 1)

  def firstBadVersion_iter(self, n):
    """
    :type n: int
    :rtype: int
    """
    low, high = 1, n
    while low <= high:
      mid = (low + high) // 2
      if self.isBadVersion(mid):
        high = mid - 1
      elif self.isBadVersion(mid + 1):
        return mid + 1
      else:
        low = mid + 1
    return low
from typing import List
class BinarySearch:
  def binary_search_iter(self, nums: List[int], target: int) -> int:
    start, end = 0, len(nums) - 1
    while start <= end:
      mid = start + (end - start) // 2
      if target == nums[mid]:
        return mid
      if target > nums[mid]:
        start = mid + 1
      else:
        end = mid - 1
    return -1

  def binary_search_rec(self, nums: List[int], target: int) -> int:
    def binary_search(start, end):
      if start > end: return -1
      mid = start +(end - start)//2
      if target == nums[mid]:
        return mid
      if target > nums[mid]:
        return binary_search(mid+1, end)
      else:
        return binary_search(start, mid-1)
    return binary_search(0, len(nums)-1)
'''
410. Split Array Largest Sum
Given an integer array nums and an integer k, split nums into k non-empty subarrays such that the largest sum of any subarray is minimized.
Return the minimized largest sum of the split.
A subarray is a contiguous part of the array.
Example 1:
Input: nums = [7,2,5,10,8], k = 2
Output: 18
Explanation: There are four ways to split nums into two subarrays.
The best way is to split it into [7,2,5] and [10,8], where the largest sum among the two subarrays is only 18.
Example 2:
Input: nums = [1,2,3,4,5], k = 2
Output: 9
Explanation: There are four ways to split nums into two subarrays.
The best way is to split it into [1,2,3] and [4,5], where the largest sum among the two subarrays is only 9.
'''
class SplitArrayLargestSum(object):
  def splitArray(self, nums, m):
    """
    :type nums: List[int]
    :type m: int
    :rtype: int
    """
    l, h = max(nums), sum(nums)
    while l < h:
      mid = (l + h) >> 1
      cnt, tmp = 1, 0
      for num in nums:
        if tmp + num > mid:
          tmp = num
          cnt += 1
        else:
          tmp += num
      if cnt > m:
        l = mid + 1
      else:
        h = mid
    return l

