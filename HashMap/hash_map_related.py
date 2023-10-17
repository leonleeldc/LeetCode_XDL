'''
560. Subarray Sum Equals K
Given an array of integers nums and an integer k, return the total number of subarrays whose sum equals to k.
A subarray is a contiguous non-empty sequence of elements within an array.
Example 1:
Input: nums = [1,1,1], k = 2
Output: 2
Example 2:
Input: nums = [1,2,3], k = 3
Output: 2
prefix sum and dictionary
'''

from typing import List
from collections import defaultdict
class HashMapRelated:
  def subarraySum(self, nums: List[int], k: int) -> int:
    acc_sum_dict = defaultdict(int, {0: 1})
    sum_to_i, total_cnt = 0, 0
    for i in range(len(nums)):
      sum_to_i += nums[i]
      if (sum_to_i - k) in acc_sum_dict:
        total_cnt += acc_sum_dict[sum_to_i - k]
      acc_sum_dict[sum_to_i] += 1
    return total_cnt
