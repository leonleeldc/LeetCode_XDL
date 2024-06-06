from typing import List
'''
260. Single Number III
Given an integer array nums, in which exactly two elements appear only once and all the other elements appear exactly twice. Find the two elements that appear only once. You can return the answer in any order.
You must write an algorithm that runs in linear runtime complexity and uses only constant extra space.
Example 1:

Input: nums = [1,2,1,3,2,5]
Output: [3,5]
Explanation:  [5, 3] is also a valid answer.
Example 2:

Input: nums = [-1,0]
Output: [-1,0]
Example 3:

Input: nums = [0,1]
Output: [1,0]
'''
class SingleNumberIII:
  def singleNumber(self, nums: List[int]) -> List[int]:
    xor = 0
    for num in nums:
      xor ^= num
    # Find rightmost set bit in xor
    rightmost_set_bit = xor & (-xor)

    # Partition the numbers into two groups and XOR within each group
    num1, num2 = 0, 0
    for num in nums:
      if num & rightmost_set_bit:
        num1 ^= num
      else:
        num2 ^= num

    return [num1, num2]
'''
137. Single Number II
Given an integer array nums where every element appears three times except for one, which appears exactly once. Find the single element and return it.
You must implement a solution with a linear runtime complexity and use only constant extra space.
Example 1:
Input: nums = [2,2,3,2]
Output: 3
Example 2:
Input: nums = [0,1,0,1,0,1,99]
Output: 99
'''
class SingleNumberII:
  def singleNumber(self, nums):
    result = 0
    for i in range(32):  # Iterate over each bit position
      bit_sum = 0
      for num in nums:
        bit_sum += (num >> i) & 1  # Count the number of '1's in this bit position
      bit_sum %= 3  # Modulo 3, as each bit in other numbers appears 3 times
      # Set the bit in 'result' if this bit is set in the single number
      if bit_sum:
        # Handle negative numbers
        if i == 31:
          result -= (1 << i)
        else:
          result |= (1 << i)
    return result
'''
136. Single Number
Given a non-empty array of integers nums, every element appears twice except for one. Find that single one.
You must implement a solution with a linear runtime complexity and use only constant extra space.
Example 1:
Input: nums = [2,2,1]
Output: 1
Example 2:
Input: nums = [4,1,2,1,2]
Output: 4
Example 3:
Input: nums = [1]
Output: 1
'''
class SingleNumber:
  def singleNumber(self, nums: List[int]) -> int:
    result = 0
    for i in range(len(nums)):
      result ^= nums[i]
    return result
'''
930. Binary Subarrays With Sum
Given a binary array nums and an integer goal, return the number of non-empty subarrays with a sum goal.
A subarray is a contiguous part of the array.

Example 1:

Input: nums = [1,0,1,0,1], goal = 2
Output: 4
Explanation: The 4 subarrays are bolded and underlined below:
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
[1,0,1,0,1]
Example 2:

Input: nums = [0,0,0,0,0], goal = 0
Output: 15
'''
class BinarySubarraysWithSum:
  def numSubarraysWithSum(self, nums, goal):
    count = 0
    running_sum = 0
    sum_counts = {0: 1}  # Initialize with 0 sum having one count

    for num in nums:
      running_sum += num
      if running_sum - goal in sum_counts:
        count += sum_counts[running_sum - goal]
      sum_counts[running_sum] = sum_counts.get(running_sum, 0) + 1

    return count


'''
1749. Maximum Absolute Sum of Any Subarray
You are given an integer array nums. The absolute sum of a subarray [numsl, numsl+1, ..., numsr-1, numsr] is abs(numsl + numsl+1 + ... + numsr-1 + numsr).
Return the maximum absolute sum of any (possibly empty) subarray of nums.
Note that abs(x) is defined as follows:
If x is a negative integer, then abs(x) = -x.
If x is a non-negative integer, then abs(x) = x.
'''
'''
1570. Dot Product of Two Sparse Vectors
Given two sparse vectors, compute their dot product.

Implement class SparseVector:

SparseVector(nums) Initializes the object with the vector nums
dotProduct(vec) Compute the dot product between the instance of SparseVector and vec
A sparse vector is a vector that has mostly zero values, you should store the sparse vector efficiently and compute the dot product between two SparseVector.

Follow up: What if only one of the vectors is sparse?

 

Example 1:

Input: nums1 = [1,0,0,2,3], nums2 = [0,3,0,4,0]
Output: 8
Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
v1.dotProduct(v2) = 1*0 + 0*3 + 0*0 + 2*4 + 3*0 = 8
Example 2:

Input: nums1 = [0,1,0,0,0], nums2 = [0,0,0,0,2]
Output: 0
Explanation: v1 = SparseVector(nums1) , v2 = SparseVector(nums2)
v1.dotProduct(v2) = 0*0 + 1*0 + 0*0 + 0*0 + 0*2 = 0
Example 3:

Input: nums1 = [0,1,0,0,2,0,0], nums2 = [1,0,0,0,3,0,4]
Output: 6
Approach 1: Non-efficient Array Approach, Time O(N), space O(1)
Approach 2: Hash Set, Time O(N) for creating HashMap, O(L) for dotProduct, space O(L) for HashMap, O(1) for dotProduct
Approach 3: Index-Value Pairs, time O(N) for creating, O(L1 + L2) for dotProduct, space O(L) for index-value pairs for non-zero values, O(1) for dotProduct
Follow-up: for non-sparse array how to optimize solution, use binary search as the indexes of non-zero elements will be sorted

'''

import bisect

from datetime import datetime

def areConsecutiveDays(date1, date2):
    delta = datetime.strptime(date2, "%Y/%m/%d") - datetime.strptime(date1, "%Y/%m/%d")
    return delta.days == 1

def consecutiveUsers(logs, N):
    # Group by user Id
    user_dates = {}
    for user, date in logs:
        if user not in user_dates:
            user_dates[user] = []
        user_dates[user].append(date)

    result = set()

    for user, dates in user_dates.items():
        dates.sort()
        consecutive_count = 1
        for i in range(1, len(dates)):
            if areConsecutiveDays(dates[i-1], dates[i]):
                consecutive_count += 1
            else:
                consecutive_count = 1

            if consecutive_count == N:
                result.add(user)
                break

    return result

# Test
logs = [('u1','2021/08/01'),('u3','2021/08/01'),('u2','2021/08/01'),('u1','2021/08/02'),('u1','2021/08/03')]
N = 3
print(consecutiveUsers(logs, N))  # Expected output: {'u1'}


class SparseVector:
  def __init__(self, nums):
    self.nums = [(i, v) for i, v in enumerate(nums) if v != 0]
  # Return the dotProduct of two sparse vectors
  def dotProduct_binarysearch(self, vec: 'SparseVector') -> int:
    # Ensure vec1 is the shorter vector for binary search
    if len(self.nums) > len(vec.nums):
      return vec.dotProduct_binarysearch(self)

    result = 0
    for i, v in self.nums:
      # Binary search for i in vec.nums
      j = bisect.bisect_left(vec.nums, (i,))
      if j < len(vec.nums) and vec.nums[j][0] == i:
        result += v * vec.nums[j][1]
    return result


class SparseVector_v2:
  def __init__(self, nums: List[int]):
    self.pairs = []
    for index, value in enumerate(nums):
      if value != 0:
        self.pairs.append([index, value])

  def dotProduct(self, vec: 'SparseVector') -> int:
    result = 0
    p, q = 0, 0
    while p < len(self.pairs) and q < len(vec.pairs):
      if self.pairs[p][0] == vec.pairs[q][0]:
        result += self.pairs[p][1] * vec.pairs[q][1]
        p += 1
        q += 1
      elif self.pairs[p][0] < vec.pairs[q][0]:
        p += 1
      else:
        q += 1
    return result

  class SparseVectorDP:
    def __init__(self, nums: List[int]):
      self.ind_vec_dict = {}
      for i, num in enumerate(nums):
        if num != 0:
          self.ind_vec_dict[i] = num

    # Return the dotProduct of two sparse vectors
    def dotProduct(self, vec: 'SparseVector') -> int:
      sum_dp = 0
      for i, v in self.ind_vec_dict.items():
        if i in vec.ind_vec_dict:
          sum_dp += v * vec.ind_vec_dict[i]
      return sum_dp


from typing import List


class SparseVectorBinarySearch:
  def __init__(self, nums: List[int]):
    self.pairs = []

    for i, num in enumerate(nums):
      if num != 0:
        self.pairs.append((i, num))
  def dotProduct(self, vec: 'SparseVector') -> int:
    if len(self.pairs) <= len(vec.pairs):
      return self._dotProduct(self, vec)
    else:
      return self._dotProduct(vec, self)
  def _dotProduct(self, small: 'SparseVector', large: 'SparseVector') -> int:
    res = 0
    for idx, val in small.pairs:
      # binary search in large.pairs for pair with index idx
      i = self._binarySearch(large.pairs, idx)
      if i >= 0:
        res += val * large.pairs[i][1]
    return res

  def _binarySearch(self, pairs: List[tuple], index: int) -> int:
    l, r = 0, len(pairs) - 1
    while l <= r:
      mid = l + (r - l) // 2
      midIdx = pairs[mid][0]
      if midIdx == index:
        return mid
      elif midIdx > index:
        r = mid - 1
      else:
        l = mid + 1
    return -1

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
