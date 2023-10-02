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

class Counting:
  '''
  70. Climbing Stairs, similarto fibonacci 509. Fibonacci Number
  '''
  def climbStairs(self, n: int) -> int:
    dp = [0] * (n + 1)
    dp[0], dp[1] = 1, 1
    for i in range(2, n + 1):
      dp[i] = dp[i - 1] + dp[i - 2]
    return dp[n]

  def climbStairs_rec(self, n:int) -> int:
    @cache
    def dp(i, n):
      if i>n: return 0
      if i==n: return 1
      return dp(i+1, n)+dp(i+2, n)
    return dp(0, n)

  def climbStairs_rec2(self, n:int) ->int:
    @cache
    def dp(n):
      if n<=1: return 1
      return dp(n-1)+dp(n-2)
    return dp(n)

  '''
  62. Unique Paths
  '''
  def uniquePaths_rec(self, m: int, n: int)->int:
    @cache
    def rec(m, n):
      if m==0 or n==0: return 1
      return rec(m-1, n)+rec(m, n-1)
    return rec(m-1, n-1)

  def uniquePaths_iter(self, m: int, n: int) ->int:
    dp = [[0] * (m+1) for _ in range(n+1)]
    dp[1][1] = 1
    for i in range(1, m+1):
      for j in range(1, n+1):
        dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m][n]

  '''
  926. Flip String to Monotone Increasing
  A binary string is monotone increasing if it consists of some number of 0's (possibly none), followed by some number of 1's (also possibly none).
You are given a binary string s. You can flip s[i] changing it from 0 to 1 or from 1 to 0.
Return the minimum number of flips to make s monotone increasing.
  '''
  # def minFlipsMonoIncr(self, s: str) -> int:

'''
134 Gas Station
There are n gas stations along a circular route, where the amount of gas at the ith station is gas[i].
You have a car with an unlimited gas tank and it costs cost[i] of gas to travel from the ith station to its next (i + 1)th station. You begin the journey with an empty tank at one of the gas stations.
Given two integer arrays gas and cost, return the starting gas station's index if you can travel around the circuit once in the clockwise direction, otherwise return -1. If there exists a solution, it is guaranteed to be unique
'''
class GasStation:
  def canCompleteCircuit_oneround(self, gas: List[int], cost: List[int]) -> int:
    start, tank, total_gas, total_cost = 0, 0, 0, 0
    for i in range(len(gas)):
      total_gas += gas[i]
      total_cost += cost[i]
      tank += gas[i] - cost[i]
      if tank < 0:
        start = i + 1
        tank = 0
    if total_gas < total_cost:
      return -1
    else:
      return start

  def canCompleteCircuit_tworound(self, gas: List[int], cost: List[int]) -> int:
    start, tank, station_visited = 0, 0, 0
    for i, (g, c) in enumerate(zip(gas+gas, cost+cost)):
      tank += g-c
      if tank < 0:
        tank = 0
        start = i+1
        station_visited = 0
      else:
        station_visited += 1
      if station_visited==len(gas):
        return start
    else:
      return -1





