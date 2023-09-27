'''
139. Word Break
Example 1:

Input: s = "leetcode", wordDict = ["leet","code"]
Output: true
Explanation: Return true because "leetcode" can be segmented as "leet code".
Example 2:

Input: s = "applepenapple", wordDict = ["apple","pen"]
Output: true
Explanation: Return true because "applepenapple" can be segmented as "apple pen apple".
Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: false
Constraints:

1 <= s.length <= 300
1 <= wordDict.length <= 1000
1 <= wordDict[i].length <= 20
s and wordDict[i] consist of only lowercase English letters.
All the strings of wordDict are unique.
'''
from typing import List
from functools import cache
class WordBreak:
  def wordBreak_iter(self, s: str, wordDict: List[str]) -> bool:
    m = len(s)
    wordDict = set(wordDict)
    dp = [True] + [False] * m
    for i in range(1, m + 1):
      for word in wordDict:
        if i >= len(word) and s[i - len(word):i] in wordDict:
          if not dp[i]: dp[i] = dp[i - len(word)]
    return dp[-1]

  def wordBreak_rec(self, s: str, wordDict: List[str]) -> bool:
    wordDict = set(wordDict)
    @cache
    def dp(s):
      if s in wordDict: return True
      for i in range(len(s)):
        if s[:i] in wordDict and dp(s[i:]):
          return True
    return dp(s)

'''
140. Word Break II https://leetcode.com/problems/word-break-ii/description/
Given a string s and a dictionary of strings wordDict, add spaces in s to construct a sentence where each word is a valid dictionary word. Return all such possible sentences in any order.
Note that the same word in the dictionary may be reused multiple times in the segmentation.
Example 1:

Input: s = "catsanddog", wordDict = ["cat","cats","and","sand","dog"]
Output: ["cats and dog","cat sand dog"]
Example 2:

Input: s = "pineapplepenapple", wordDict = ["apple","pen","applepen","pine","pineapple"]
Output: ["pine apple pen apple","pineapple pen apple","pine applepen apple"]
Explanation: Note that you are allowed to reuse a dictionary word.
Example 3:

Input: s = "catsandog", wordDict = ["cats","dog","sand","and","cat"]
Output: []
 

Constraints:

1 <= s.length <= 20
1 <= wordDict.length <= 1000
1 <= wordDict[i].length <= 10
s and wordDict[i] consist of only lowercase English letters.
All the strings of wordDict are unique.
Input is generated in a way that the length of the answer doesn't exceed 105.
'''
from collections import Counter
import sys, os
from functools import lru_cache, partial
class WordBreakII:
  def wordBreak_iter(self, s: str, wordDict: List[str]) -> List[str]:
    if set(Counter(s).keys()) > set(Counter(''.join(wordDict)).keys()): return []
    word_set = set(wordDict)
    dp = [[] for _ in range(len(s) + 1)]
    dp[0] = ['']
    for end in range(1, len(s) + 1):
      sublist = []
      for begin in range(end):
        if s[begin:end] in word_set:
          for subsent in dp[begin]:
            sublist.append((subsent + ' ' + s[begin:end]).strip())
      dp[end] = sublist
    return dp[-1]

  def wordBreak_rec(self, s: str, wordDict: List[str]) -> List[str]:
    if set(Counter(s).keys()) > set(Counter(''.join(wordDict)).keys()): return [] # check for impossible cases
    wordSet = set(wordDict)
    if set(Counter(s).keys()) > set(Counter(''.join(wordDict)).keys()): return []
    wordSet = set(wordDict)
    @cache
    def dp(start: int) -> List[str]:
      if start == len(s): return ['']
      sentences = []
      for end in range(start + 1, len(s) + 1):
        if s[start:end] in wordSet:
          for sentence in dp(end):
            if sentence:
              sentences.append(s[start:end] + ' ' + sentence)
            else:
              sentences.append(s[start:end])
      return sentences
    return dp(0)

class StickerToSpellWord:
  def minStickers(self, stickers: List[str], target: str) -> int:
    present = {}
    # present maps index of target to character
    # for example for target='that' it will be as follows-
    # present={0:'t', 1:'h', 2:'a', 3:'t'}
    for i, c in enumerate(target):
      present[i] = c
    n = len(stickers)
    m = len(target)
    @lru_cache(None)
    def dp(idx, mask):
      if idx == n:
        flag = True
        for i in range(m):
          if ((1 << i) & mask) != 0:
            flag = False
            break
        if flag: return 0
        return sys.maxsize
      ans = sys.maxsize
      Flag = False
      temp_mask = mask
      # check and use this idx
      count = Counter(stickers[idx])
      for i in range(m):
        if ((1 << i) & mask) != 0 and count[present[i]] > 0:
          mask ^= (1 << i)
          count[present[i]] -= 1
          Flag = True
      if Flag:
        ans = min(ans, 1 + dp(idx, mask))
      # dont use this idx
      ans = min(ans, dp(idx + 1, temp_mask))
      return ans

    res = dp(0, (1 << m) - 1)
    return res if res < sys.maxsize else -1

