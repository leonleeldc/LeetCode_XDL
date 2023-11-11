from typing import List
class MaxConsecutiveOnes:
  '''
  485. Max Consecutive Ones
  Given a binary array nums, return the maximum number of consecutive 1's in the array.
  Example 1:

Input: nums = [1,1,0,1,1,1]
Output: 3
Explanation: The first two digits or the last three digits are consecutive 1s. The maximum number of consecutive 1s is 3.
Example 2:

Input: nums = [1,0,1,1,0,1]
Output: 2


Constraints:

1 <= nums.length <= 105
nums[i] is either 0 or 1.
  '''
  def findMaxConsecutiveOnes(self, nums: List[int]) -> int:
    max_ones, cur_ones = 0, 0
    for i, num in enumerate(nums):
      if num == 1:
        cur_ones += 1
      else:
        max_ones = max(max_ones, cur_ones)
        cur_ones = 0
    max_ones = max(max_ones, cur_ones)
    return max_ones
  '''
  487. Max Consecutive Ones II
  Given a binary array nums, return the maximum number of consecutive 1's in the array if you can flip at most one 0.
Example 1:

Input: nums = [1,0,1,1,0]
Output: 4
Explanation: 
- If we flip the first zero, nums becomes [1,1,1,1,0] and we have 4 consecutive ones.
- If we flip the second zero, nums becomes [1,0,1,1,1] and we have 3 consecutive ones.
The max number of consecutive ones is 4.
Example 2:

Input: nums = [1,0,1,1,0,1]
Output: 4
Explanation: 
- If we flip the first zero, nums becomes [1,1,1,1,0,1] and we have 4 consecutive ones.
- If we flip the second zero, nums becomes [1,0,1,1,1,1] and we have 4 consecutive ones.
The max number of consecutive ones is 4.

  '''
  def findMaxConsecutiveOnes_II(self, nums: List[int]) -> int:
    counter, cur_ones, prev_ones = 0, 0, 0
    for i, num in enumerate(nums):
      counter += 1
      if num == 0:
        prev_ones = counter
        counter = 0
      if cur_ones < (prev_ones + counter):
        cur_ones = prev_ones + counter
    return cur_ones
'''
1004. Max Consecutive Ones III
Given a binary array nums and an integer k, return the maximum number of consecutive 1's in the array if you can flip at most k 0's.
Example 1:

Input: nums = [1,1,1,0,0,0,1,1,1,1,0], k = 2
Output: 6
Explanation: [1,1,1,0,0,1,1,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
Example 2:

Input: nums = [0,0,1,1,0,0,1,1,1,0,1,1,0,0,0,1,1,1,1], k = 3
Output: 10
Explanation: [0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,1,1,1,1]
Bolded numbers were flipped from 0 to 1. The longest subarray is underlined.
'''
class MaxConsecutiveOnesIII:
  def longestOnes_optimized(self, nums: List[int], k: int) -> int:
    left = 0
    for right in range(len(nums)):
      k -= 1 - nums[right]
      if k < 0:
        k += 1 - nums[left]
        left += 1
    return right - left + 1
  def longestOnes_patternized(self, nums: List[int], k: int) -> int:
    left = max_len = zeros_flipped = 0  # Track the number of zeros that have been flipped
    for right in range(len(nums)):
      # If we encounter a 0, increment the count of flipped zeros
      if nums[right] == 0: zeros_flipped += 1
      # If we have flipped more than k zeros, move the left pointer
      # to the right until we have k or fewer zeros in the window
      while zeros_flipped > k:
        if nums[left] == 0:
          zeros_flipped -= 1
        left += 1
      # The current window size is right - left + 1, check if it's the largest we've seen
      max_len = max(max_len, right - left + 1)
    return max_len