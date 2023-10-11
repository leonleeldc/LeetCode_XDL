'''
1749. Maximum Absolute Sum of Any Subarray
You are given an integer array nums. The absolute sum of a subarray [numsl, numsl+1, ..., numsr-1, numsr] is abs(numsl + numsl+1 + ... + numsr-1 + numsr).
Return the maximum absolute sum of any (possibly empty) subarray of nums.
Note that abs(x) is defined as follows:
If x is a negative integer, then abs(x) = -x.
If x is a non-negative integer, then abs(x) = x.
'''
from typing import List
class MaxAbsSumSubArr:
  def maxAbsoluteSumIter(self, nums: List[int]) -> int:
    min_sum, max_sum, max_curr, min_curr = 0, 0, 0, 0
    for num in nums:
      max_curr = max(num, max_curr + num)
      max_sum = max(max_sum, max_curr)
      min_curr = min(num, min_curr + num)
      min_sum = min(min_sum, min_curr)
    return max(max_sum, abs(min_sum))
  def maxAbsoluteSumRec(self, nums: List[int]) -> int:
    def rec(index, max_curr, min_curr, max_sum, min_sum):
      # Base case: if we've gone past the end of the array,
      # return the max of max_sum and abs(min_sum)
      if index == len(nums): return max(max_sum, abs(min_sum))
      # Update max and min current subarray sum
      max_curr = max(nums[index], max_curr + nums[index])
      min_curr = min(nums[index], min_curr + nums[index])
      # Update max and min total subarray sum
      max_sum = max(max_sum, max_curr)
      min_sum = min(min_sum, min_curr)
      # Recursive call to the next index
      return rec(index + 1, max_curr, min_curr, max_sum, min_sum)
    # Initial call to the helper function with starting parameters
    return rec(0, 0, 0, 0, 0)

class ShortestSubArrMaxsum:
  def shortest_subarray_with_max_sum(self, arr):
    if not arr: return 0
    max_sum = max(arr)
    max_sum_length = 1  # A single element is the shortest possible subarray
    current_sum = 0
    start_index = 0  # To keep track of the start of the current subarray
    for end_index in range(len(arr)):
      current_sum += arr[end_index]
      # If the current_sum becomes negative, move the start_index
      while current_sum < 0 and start_index <= end_index:
        current_sum -= arr[start_index]
        start_index += 1
      # Check if current_sum is now the new max
      if current_sum > max_sum:
        max_sum = current_sum
        max_sum_length = end_index - start_index + 1
      elif current_sum == max_sum:
        max_sum_length = min(max_sum_length, end_index - start_index + 1)
      # If current_sum is larger than max, try to minimize length
      while current_sum - arr[start_index] >= max_sum and start_index <= end_index:
        current_sum -= arr[start_index]
        start_index += 1
        max_sum_length = min(max_sum_length, end_index - start_index + 1)
    return max_sum_length


import sys

class NSum():
  # Function to return the sum of a
  # triplet which is closest to x
  def solution(self, arr, x):
    # Sort the array
    arr.sort();
    # To store the closest sum
    closestSum = sys.maxsize;
    # Fix the smallest number among
    # the three integers
    for i in range(len(arr) - 2):
      # Two pointers initially pointing at
      # the last and the element
      # next to the fixed element
      ptr1 = i + 1;
      ptr2 = len(arr) - 1;
      # While there could be more pairs to check
      while (ptr1 < ptr2):
        # Calculate the sum of the current triplet
        sum = arr[i] + arr[ptr1] + arr[ptr2];
        # If the sum is more closer than
        # the current closest sum
        if (abs(x - sum) < abs(x - closestSum)):
          closestSum = sum;
          # If sum is greater than x then decrement
        # the second pointer to get a smaller sum
        if (sum > x):
          ptr2 -= 1;
          # Else increment the first pointer
        # to get a larger sum
        else:
          ptr1 += 1;
          # Return the closest sum found
    return closestSum;
