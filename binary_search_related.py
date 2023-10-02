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

