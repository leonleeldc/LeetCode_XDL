'''
1. Two Sum
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
'''
from typing import List
from collections import defaultdict
class SumSeries:
  def twoSum_2run(self, nums: List[int], target: int) -> List[int]:
    val_dict = defaultdict(int)
    for i, num in enumerate(nums):
      if num not in val_dict:
        val_dict[num] = i
    for i, num in enumerate(nums):
      if target - num in val_dict and i != val_dict[target - num]:
        return i, val_dict[target - num]
  def twoSum_1run(self, nums: List[int], target: int) -> List[int]:
    hash_set = {}
    for i, num in enumerate(nums):
      if target - num in hash_set and hash_set[target - num] != i:
        return [i, hash_set[target - num]]
      hash_set[num] = i
  '''
  15 3sum
  Given an integer array nums, return all the triplets [nums[i], nums[j], nums[k]] such that i != j, i != k, and j != k, and nums[i] + nums[j] + nums[k] == 0.
  Notice that the solution set must not contain duplicate triplets.
  '''
  def threeSum(self, nums: List[int]) -> List[List[int]]:
    nums.sort()
    output = set()
    for i, num in enumerate(nums):
      j, k = i + 1, len(nums) - 1
      while j < k:
        if nums[j] + nums[k] == -num:
          output.add((num, nums[j], nums[k]))
          j += 1
          k -= 1
        elif nums[j] + nums[k] < -num:
          j += 1
        else:
          k -= 1
    return output
  '''
  3sum variations with a^2 + b^2 = c^2 (2)?
  '''
  def threeSumSquared(self, nums):
    # First, let's square each number and sort the list
    nums = sorted([x ** 2 for x in nums])
    res = []
    n = len(nums)
    for i in range(n - 1, 1, -1): # Let's use c as our third number
      c = nums[i]
      l, r = 0, i - 1 # Now, we'll use two pointers to find a and b
      while l < r:
        if nums[l] + nums[r] == c:
          a = int(nums[l] ** 0.5)
          b = int(nums[r] ** 0.5)
          c_sqrt = int(c ** 0.5)
          res.append([a, b, c_sqrt])
          l += 1
          r -= 1
        elif nums[l] + nums[r] < c:
          l += 1
        else:
          r -= 1
    return res
