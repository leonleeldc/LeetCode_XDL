from typing import List
from collections import defaultdict, Counter
'''
992. Subarrays with K Different Integers
Solved
Hard
Topics
Companies
Hint
Given an integer array nums and an integer k, return the number of good subarrays of nums.

A good array is an array where the number of different integers in that array is exactly k.

For example, [1,2,3,1,2] has 3 different integers: 1, 2, and 3.
A subarray is a contiguous part of an array.

 

Example 1:

Input: nums = [1,2,1,2,3], k = 2
Output: 7
Explanation: Subarrays formed with exactly 2 different integers: [1,2], [2,1], [1,2], [2,3], [1,2,1], [2,1,2], [1,2,1,2]
Example 2:

Input: nums = [1,2,1,3,4], k = 3
Output: 3
Explanation: Subarrays formed with exactly 3 different integers: [1,2,1,3], [2,1,3], [1,3,4].
The following approach is quite smart! We need to dive deep to understand why it is correct !!!
'''
class SubWithKDiffInts:
    def subarraysWithKDistinct(self, nums: List[int], k: int) -> int:
        def at_most_k_distinct(k: int) -> int:
            count = defaultdict(int)
            i = 0
            result = 0
            for j in range(len(nums)):
                if count[nums[j]] == 0:
                    k -= 1
                count[nums[j]] += 1
                while k < 0:
                    count[nums[i]] -= 1
                    if count[nums[i]] == 0:
                        k += 1
                    i += 1
                result += j - i + 1
            return result

        return at_most_k_distinct(k) - at_most_k_distinct(k - 1)
'''
283. Move Zeros
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements.
Note that you must do this in-place without making a copy of the array.
Example 1:
Input: nums = [0,1,0,3,12]
Output: [1,3,12,0,0]
Example 2:
Input: nums = [0]
Output: [0]
'''
class MoveZeros:
  def moveZeroes(self, nums: List[int]) -> None:
    """
    Do not return anything, modify nums in-place instead.
    """
    count = 0
    for i in range(len(nums)):
      if nums[i] != 0:
        nums[count], nums[i] = nums[i], nums[count]
        count += 1
    '''
    add difficulty: there are k zeors between two non_zero elements
    '''
  def moveZeroesWithGap(self, nums, k):
    write_index = 0  # Initialize the write pointer
    n = len(nums)
    # Loop through all elements with the read pointer
    for read_index in range(n):
      # When we find a non-zero element, we move it to the 'write_index' position
      if nums[read_index] != 0:
        # Place the non-zero element at 'write_index'
        nums[write_index] = nums[read_index]
        # Fill the next 'k' positions with zeros, if they are within the bounds of the array
        next_index = write_index + 1
        for _ in range(k):
          if next_index < n:
            nums[next_index] = 0
          next_index += 1
        # Skip 'k' positions ahead
        write_index += k + 1
    # Fill the rest of the array with zeros if 'write_index' is within the bounds
    while write_index < n:
      nums[write_index] = 0
      write_index += 1

'''
1838. Frequency of the Most Frequent Element
The frequency of an element is the number of times it occurs in an array.
You are given an integer array nums and an integer k. In one operation, you can choose an index of nums and increment the element at that index by 1.
Return the maximum possible frequency of an element after performing at most k operations.
Example 1:

Input: nums = [1,2,4], k = 5
Output: 3
Explanation: Increment the first element three times and the second element two times to make nums = [4,4,4].
4 has a frequency of 3.
Example 2:

Input: nums = [1,4,8,13], k = 5
Output: 2
Explanation: There are multiple optimal solutions:
- Increment the first element three times to make nums = [4,4,8,13]. 4 has a frequency of 2.
- Increment the second element four times to make nums = [1,8,8,13]. 8 has a frequency of 2.
- Increment the third element five times to make nums = [1,4,13,13]. 13 has a frequency of 2.
Example 3:

Input: nums = [3,9,6], k = 2
Output: 1
'''
class FreqMostFreqElements:
  def maxFrequency(self, nums: List[int], k: int) -> int:
    nums.sort()
    l, summ, ans = 0, 0, 0
    for r in range(len(nums)):
      summ += nums[r]
      while l < r and summ + k < nums[r] * (r - l + 1):
        summ -= nums[l]
        l += 1
      ans = max(ans, r - l + 1)
    return ans
'''
76. Minimum Window Substring
Given two strings s and t of lengths m and n respectively, return the minimum window 
substring
 of s such that every character in t (including duplicates) is included in the window. If there is no such substring, return the empty string "".
The testcases will be generated such that the answer is unique.
Example 1:

Input: s = "ADOBECODEBANC", t = "ABC"
Output: "BANC"
Explanation: The minimum window substring "BANC" includes 'A', 'B', and 'C' from string t.
Example 2:

Input: s = "a", t = "a"
Output: "a"
Explanation: The entire string s is the minimum window.
Example 3:

Input: s = "a", t = "aa"
Output: ""
Explanation: Both 'a's from t must be included in the window.
Since the largest window of s only has one 'a', return empty string.
'''
class MinimumWindowSubstring:
  def minWindow_moreefficient(self, s: str, t: str) -> str:
    m, n = len(s), len(t)
    if n > m: return ''
    counter = Counter(t)
    start = 0
    min_slice = (0, m + 1)
    remaining_chars = n
    for i in range(m):
      if counter[s[i]] > 0: remaining_chars -= 1
      counter[s[i]] -= 1
      while start < i and counter[s[start]] < 0:
        counter[s[start]] += 1
        start += 1
      if remaining_chars == 0:
        if min_slice[1] - min_slice[0] > i - start:
          min_slice = (start, i + 1)
    return '' if min_slice[1] == m + 1 else s[slice(*min_slice)]
  def minWindow(self, source: str, target: str) -> str:
    s, t = Counter(source), Counter(target)
    if any(s[i] < t[i] for i in source): return ''
    if any(s[i] == 0 for i in t): return ''
    res, count, j = source, defaultdict(int), 0
    for i in range(len(source)):
      count[source[i]] += 1
      while j <= i and all(t[x] <= count[x] for x in t):
        res = source[j:i + 1] if i - j + 1 < len(res) else res
        count[source[j]] -= 1
        j += 1
    return res
  '''
  the following function follows template more
  '''
  def minWindow_template(self, source: str, target: str) -> str:
    if not source or not target: return ''
    counter_t = Counter(target)
    l, r, min_len = 0, 0, len(source)
    sind_dict = defaultdict(int)
    ans = float('inf'), None, None
    formed = 0
    while r < len(source):
      ch = source[r]
      sind_dict[ch] += 1
      if ch in counter_t and sind_dict[ch] == counter_t[ch]:
        formed += 1
      while l <= r and formed == len(counter_t):
        ch = source[l]
        if r - l + 1 < ans[0]:
          ans = (r - l + 1, l, r)
        sind_dict[ch] -= 1
        if ch in counter_t and sind_dict[ch] < counter_t[ch]:
          formed -= 1
        l += 1
      r += 1
    return '' if ans[0] == float('inf') else source[ans[1]:ans[2] + 1]

'''
340. Longest Substring with At Most K Distinct
Given a string s and an integer k, return the length of the longest 
substring
 of s that contains at most k distinct characters.

 

Example 1:

Input: s = "eceba", k = 2
Output: 3
Explanation: The substring is "ece" with length 3.
Example 2:

Input: s = "aa", k = 1
Output: 2
Explanation: The substring is "aa" with length 2.
 

Constraints:

1 <= s.length <= 5 * 104
0 <= k <= 50
'''


class LongestSubstringWithKDistinct:
  def lengthOfLongestSubstringKDistinct(self, s: str, k: int) -> int:
    n = len(s)
    if n * k == 0: return 0
    ch_ind_dict = defaultdict(int)
    l, r, max_return = 0, 0, 0
    while r < n:
      ch_ind_dict[s[r]] = r
      r += 1
      if len(ch_ind_dict) == k + 1:
        del_ind = min(ch_ind_dict.values())
        del ch_ind_dict[s[del_ind]]
        l = del_ind + 1
      max_return = max(max_return, r - l)
    return max_return

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