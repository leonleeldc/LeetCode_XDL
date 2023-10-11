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