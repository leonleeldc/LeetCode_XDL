from collections import Counter
from typing import List
class StringProblemClass:
  '''
    443. String Compression
Given an array of characters chars, compress it using the following algorithm:
Begin with an empty string s. For each group of consecutive repeating characters in chars:
If the group's length is 1, append the character to s.
Otherwise, append the character followed by the group's length.
The compressed string s should not be returned separately, but instead, be stored in the input character array chars. Note that group lengths that are 10 or longer will be split into multiple characters in chars.
After you are done modifying the input array, return the new length of the array.
You must write an algorithm that uses only constant extra space.
Example 1:

Input: chars = ["a","a","b","b","c","c","c"]
Output: Return 6, and the first 6 characters of the input array should be: ["a","2","b","2","c","3"]
Explanation: The groups are "aa", "bb", and "ccc". This compresses to "a2b2c3".
Example 2:

Input: chars = ["a"]
Output: Return 1, and the first character of the input array should be: ["a"]
Explanation: The only group is "a", which remains uncompressed since it's a single character.
Example 3:

Input: chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
Output: Return 4, and the first 4 characters of the input array should be: ["a","b","1","2"].
Explanation: The groups are "a" and "bbbbbbbbbbbb". This compresses to "ab12".

  '''
  def compress_three_variable(self, chars: List[str]) -> int:
    first, second, third = 0, 0, 0
    while third < len(chars):
      while chars[second] == chars[third]:
        third += 1
        if third >= len(chars):
          break
      chars[first] = chars[second]
      first += 1
      ls = list(str(third - second))
      if third - second > 1:
        for n in ls:
          chars[first] = n
          first += 1
      second = third
    return first
  def compress_two_variable(self, chars: List[str]) -> int:
    write_idx = read_idx = 0
    while read_idx < len(chars):
      char, length = chars[read_idx], 1
      # Move read_idx to the end of the group of repeating characters
      while (read_idx + 1 < len(chars)) and (chars[read_idx] == chars[read_idx + 1]):
        read_idx += 1
        length += 1
      # Write the character to the new position
      chars[write_idx] = char
      write_idx += 1
      # If the group length is greater than 1, write the length of the group
      if length > 1:
        for digit in str(length):
          chars[write_idx] = digit
          write_idx += 1
      read_idx += 1
    return write_idx
  '''
  3. Longest Substring Without Repeating Characters
  Example 1:

Input: s = "abcabcbb"
Output: 3
Explanation: The answer is "abc", with the length of 3.
Example 2:

Input: s = "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: s = "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3.
Notice that the answer must be a substring, "pwke" is a subsequence and not a substring.
  '''

  def lengthOfLongestSubstring_v1(self, s: str) -> int:
    mp = {}
    res, left = 0, 0
    for right in range(len(s)):
      if s[right] in mp:
        left = max(mp[s[right]], left)
      res = max(res, right - left + 1)
      mp[s[right]] = right + 1
    return res

  def lengthOfLongestSubstring_v2(self, s: str) -> int:
    if not s: return 0
    l, r, max_len = 0, 1, 1
    while r < len(s):
      if s[r] in s[l:r]:
        l = l + s[l:r].index(s[r]) + 1
      max_len = max(max_len, r - l + 1)
      r += 1
    return max_len
  def shortest_substring(self, s, chars):
    # Count the occurrences of characters in `chars`
    char_count = Counter(chars)
    # Count the occurrences of characters in the current window
    window_count = Counter()
    # Number of unique characters matched
    formed = 0
    # Two pointers for the sliding window
    left, right = 0, 0
    # Length and start index of the minimum window
    min_len, min_start = float('inf'), 0
    while right < len(s):
      # Add character from the right to the window
      if s[right] in char_count:
        window_count[s[right]] += 1
        if window_count[s[right]] == char_count[s[right]]:
          formed += 1
      # Remove characters from the left of the window
      while left <= right and formed == len(char_count):
        # Update the minimum window's length and starting index
        if right - left + 1 < min_len:
          min_len = right - left + 1
          min_start = left
        # Move the left pointer
        if s[left] in char_count:
          window_count[s[left]] -= 1
          if window_count[s[left]] < char_count[s[left]]:
            formed -= 1
        left += 1
      # Move the right pointer
      right += 1
    # Return the minimum window
    return "" if min_len == float('inf') else s[min_start:min_start + min_len]
