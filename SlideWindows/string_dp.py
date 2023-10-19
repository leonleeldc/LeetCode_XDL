'''
3. Longest Substring Without Repeating Characters
'''
class LongestSubstrNoRepchar:
  def lengthOfLongestSubstring(self, s: str) -> int:
    if not s: return 0
    l, r, max_len = 0, 1, 1
    while r < len(s):
      if s[r] in s[l:r]:
        l = l + s[l:r].index(s[r]) + 1
      max_len = max(max_len, r - l + 1)
      r += 1
    return max_len

  def lengthOfLongestSubstring_slidewindow(self, s: str) -> int:
    '''
    Sliding Window Optimized
    '''
    mp = {}
    res, left = 0, 0
    for right in range(len(s)):
      if s[right] in mp:
        left = max(mp[s[right]], left)
      res = max(res, right - left + 1)
      mp[s[right]] = right + 1
    return res
'''
To find a 'SuperArray' subarray that satisfies the conditions, you can use the sliding window approach.
'''
class SuperArraySubArray:
  def super_array_subarray(self, k, arr):
    n = len(arr)
    # Check if the array length is a power of 2
    if n & (n - 1) != 0:
      return []
    # Compute the prefix sum array
    prefix_sum = [0] * (n + 1)
    for i in range(n):
      prefix_sum[i + 1] = prefix_sum[i] + arr[i]
    left, right = 0, 0
    while right <= n:
      current_sum = prefix_sum[right] - prefix_sum[left]
      # If the current sum lies in the interval [k, 2k)
      if k <= current_sum < 2 * k:
        # Check if the subarray length is a power of 2
        if (right - left) & (right - left - 1) == 0:
          return arr[left:right]
        # If not a 'SuperArray', expand the window to the right
        right += 1
      # If the current sum is below k, expand the window to the right
      elif current_sum < k:
        right += 1
      # If the current sum is >= 2k, shrink the window from the left
      else:
        left += 1
    return []
