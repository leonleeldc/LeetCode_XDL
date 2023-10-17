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
import collections
'''
1320. Minimum Distance to Type a Word Using Two Fingers
You have a keyboard layout as shown above in the X-Y plane, where each English uppercase letter is located at some coordinate.

For example, the letter 'A' is located at coordinate (0, 0), the letter 'B' is located at coordinate (0, 1), the letter 'P' is located at coordinate (2, 3) and the letter 'Z' is located at coordinate (4, 1).
Given the string word, return the minimum total distance to type such string using only two fingers.

The distance between coordinates (x1, y1) and (x2, y2) is |x1 - x2| + |y1 - y2|.

Note that the initial positions of your two fingers are considered free so do not count towards your total distance, also your two fingers do not have to start at the first letter or the first two letters.
Input: word = "CAKE"
Output: 3
Explanation: Using two fingers, one optimal way to type "CAKE" is: 
Finger 1 on letter 'C' -> cost = 0 
Finger 1 on letter 'A' -> cost = Distance from letter 'C' to letter 'A' = 2 
Finger 2 on letter 'K' -> cost = 0 
Finger 2 on letter 'E' -> cost = Distance from letter 'K' to letter 'E' = 1 
Total distance = 3
Example 2:

Input: word = "HAPPY"
Output: 6
Explanation: Using two fingers, one optimal way to type "HAPPY" is:
Finger 1 on letter 'H' -> cost = 0
Finger 1 on letter 'A' -> cost = Distance from letter 'H' to letter 'A' = 2
Finger 2 on letter 'P' -> cost = 0
Finger 2 on letter 'P' -> cost = Distance from letter 'P' to letter 'P' = 0
Finger 1 on letter 'Y' -> cost = Distance from letter 'A' to letter 'Y' = 4
Total distance = 6
Top down: O(n*27^2)
'''
class MinDistTypingFingers:
  def minimumDistance_rec_3d(self, word: str) -> int:
    kRest = 26  # Represents a state where a finger is not on any letter
    n = len(word)
    mem = [[[-1] * 27 for _ in range(27)] for _ in range(n)] # Initialize memoization table
    def cost(c1, c2): # Cost function to move from character c1 to c2
      if c1 == kRest: return 0
      return abs(c1 // 6 - c2 // 6) + abs(c1 % 6 - c2 % 6)
    def dp(i, l, r): # Recursive DP function
      if i == n: return 0
      if mem[i][l][r] >= 0: return mem[i][l][r]
      c = ord(word[i]) - ord('A')
      mem[i][l][r] = min(dp(i + 1, c, r) + cost(l, c), dp(i + 1, l, c) + cost(r, c))
      return mem[i][l][r]
    return dp(0, kRest, kRest) # Call the DP function starting from the first character with both fingers resting
  def minimumDistance_rec_2d(self, word: str) -> int:
      kRest = 26  # Represents a state where a finger is not on any letter
      n = len(word)
      # Cost function to move from character c1 to c2
      def cost(c1, c2):
          if c1 == kRest:
              return 0
          return abs(c1 // 6 - c2 // 6) + abs(c1 % 6 - c2 % 6)
      # Recursive DP function
      @cache
      def dp(i, o):
          if i == n: return 0
          p = kRest if i == 0 else ord(word[i - 1]) - ord('A')
          c = ord(word[i]) - ord('A')
          return min(dp(i + 1, o) + cost(p, c),  # same finger
                          dp(i + 1, p) + cost(o, c))  # other finger
      # Call the DP function starting from the first character
      # with the other finger resting
      return dp(0, kRest)
  def minimumDistance_iter(self, word: str) -> int:
    kRest = 26  # Represents a state where a finger is not on any letter
    n = len(word)
    # Initialize DP table
    dp = [[float('inf')] * 27 for _ in range(n + 1)]
    dp[0][kRest] = 0
    # Cost function to move from character c1 to c2
    def cost(c1, c2):
      if c1 == kRest:
        return 0
      return abs(c1 // 6 - c2 // 6) + abs(c1 % 6 - c2 % 6)
    # Iterative DP
    for i in range(n):
      p = kRest if i == 0 else ord(word[i - 1]) - ord('A')
      c = ord(word[i]) - ord('A')
      for j in range(27):  # 26 letters + kRest
        dp[i + 1][j] = min(dp[i + 1][j], dp[i][j] + cost(p, c))  # same finger
        dp[i + 1][p] = min(dp[i + 1][p], dp[i][j] + cost(j, c))  # other finger
    # Find and return the minimum cost
    return min(dp[n])


'''
135. Candy
There are n children standing in a line. Each child is assigned a rating value given in the integer array ratings.

You are giving candies to these children subjected to the following requirements:

Each child must have at least one candy.
Children with a higher rating get more candies than their neighbors.
Return the minimum number of candies you need to have to distribute the candies to the children.

 

Example 1:

Input: ratings = [1,0,2]
Output: 5
Explanation: You can allocate to the first, second and third child with 2, 1, 2 candies respectively.
Example 2:

Input: ratings = [1,2,2]
Output: 4
Explanation: You can allocate to the first, second and third child with 1, 2, 1 candies respectively.
The third child gets 1 candy because it satisfies the above two conditions.
'''


class CandyGivenByRating:
  def candy(self, ratings: List[int]) -> int:
    '''
    [4,5,3,1,0,2,8]
    [4,5,3,1,0,2,8,9]
    [4,5,3,1,0,2,8,9,9,9]
    [4,5,3,1,0,2,8,9,9,9,10]
    [4,5]
    [1,0,2]
    '''
    dp = [1] * len(ratings)
    for i in range(1, len(ratings)):
      if ratings[i] > ratings[i - 1]:
        dp[i] = max(dp[i - 1] + 1, dp[i])
    for i in range(len(ratings) - 2, -1, -1):
      if ratings[i] > ratings[i + 1]:
        dp[i] = max(dp[i + 1] + 1, dp[i])
    return sum(dp)

'''
1458. Max Dot Product of Two Subsequences
Given two arrays nums1 and nums2.

Return the maximum dot product between non-empty subsequences of nums1 and nums2 with the same length.

A subsequence of a array is a new array which is formed from the original array by deleting some (can be none) of the characters without disturbing the relative positions of the remaining characters. (ie, [2,3,5] is a subsequence of [1,2,3,4,5] while [1,5,3] is not).
'''
class MaxDotProductTwoSubseq:
  def maxDotProduct(self, nums1: List[int], nums2: List[int]) -> int:
    m, n = len(nums1), len(nums2)
    dp = [[-float('inf')] * (n + 1) for _ in range(m + 1)]
    # Build dp table based on the recursive relation
    for i in range(1, m + 1):
      for j in range(1, n + 1):
        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1] + nums1[i - 1] * nums2[j - 1],
                       nums1[i - 1] * nums2[j - 1])
    return dp[m][n]
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

You are given an integer array nums and an integer k. You can partition the array into at most k non-empty adjacent subarrays. The score of a partition is the sum of the averages of each subarray.
Note that the partition must use every integer in nums, and that the score is not necessarily an integer.
Return the maximum score you can achieve of all the possible partitions. Answers within 10-6 of the actual answer will be accepted.
'''
class LargSumAve:
  def largestSumOfAverages_iter(self, nums: List[int], k: int) -> float:
    n = len(nums)
    dp = [[0] * (k + 1) for _ in range(n + 1)]  # Initialize the dp array
    prefix_sum = [0] * (n + 1)
    for i in range(n):
      prefix_sum[i + 1] = prefix_sum[i] + nums[i]
      dp[i + 1][1] = prefix_sum[i + 1] / (i + 1)
    for i in range(1, n + 1):
      for kk in range(2, k + 1):
        for j in range(1, i):
          dp[i][kk] = max(dp[i][kk], dp[j][kk - 1] + (prefix_sum[j] - prefix_sum[i]) / (j - i))
    return dp[n][k]
  def largestSumOfAverages_rec(self, nums: List[int], k: int) -> float:
    n = len(nums)
    prefix_sum = [0] * (n + 1)
    for i in range(n):
      prefix_sum[i + 1] = prefix_sum[i] + nums[i]
    @cache
    def dp(n, k):
      if k==1: return prefix_sum[n]/n
      max_sum = 0
      for i in range(k-1, n):
         max_sum = max(max_sum, dp(i, k-1)+(prefix_sum[n]-prefix_sum[i])/(n-i))
      return max_sum
    return dp(n, k)

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
  class Solution:
    def minStickers(self, stickers: List[str], target: str) -> int:
      t_count = Counter(target)
      s_count = [Counter(sticker) & t_count for sticker in stickers if any(ch in t_count for ch in sticker)] # Filter & Modify stickers
      i = 0 # Remove dominated stickers
      while i < len(s_count):
        if any(all(s_count[i][ch] <= s_count[j][ch] for ch in s_count[i]) for j in range(len(s_count)) if i != j):
          s_count.pop(i)
        else: i += 1
      # Convert to list and sort by frequency in target
      s_count = [list(sticker.elements()) for sticker in s_count]
      for sticker in s_count:
        sticker.sort(key=lambda ch: t_count[ch], reverse=True)
      dp = [float('inf')] * (1 << len(target)) # DP with bitmask representation
      dp[0] = 0  # base case
      for mask in range(1, len(dp)):
        sub_target = [target[i] for i in range(len(target)) if mask & (1 << i)]
        if not any(ch in sub_target for sticker in s_count for ch in sticker): continue # Early termination if impossible
        for sticker in s_count:
          if sub_target[0] not in sticker: continue           # Skip if sticker does not contain the first char of sub_target
          rem_mask = mask  # Try to use the sticker and update the mask
          for ch in sticker:
            for i, tc in enumerate(target): # Find the index of the character to unset
              if (rem_mask >> i) & 1 and tc == ch:
                rem_mask ^= (1 << i)
                break
            if rem_mask == 0: break  # Early break if all chars satisfied
          dp[mask] = min(dp[mask], 1 + dp[rem_mask]) # Update DP
      return dp[-1] if dp[-1] != float('inf') else -1

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

  def findPaths(self, n):
    def dfs(x, y, path):
      # If the position is out of the bounds or already visited, return
      if x >= n or y >= n: return
      # If the position is the bottom-right corner, add the path and return
      if x == n - 1 and y == n - 1:
        paths.append(path)
        return
      # Move right and down recursively, appending to path
      dfs(x + 1, y, path + [(x + 1, y)])  # Move right
      dfs(x, y + 1, path + [(x, y + 1)])  # Move down
    paths = []
    dfs(0, 0, [(0, 0)])
    return paths

  # Example
  n = 3
  print(findPaths(n))

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





