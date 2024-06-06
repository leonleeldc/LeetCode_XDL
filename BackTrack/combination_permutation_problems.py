'''
39. Combination Sum
Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.
The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the
frequency
 of at least one of the chosen numbers is different.

'''
from typing import List
from functools import cache


class CombinationSumSeries:
  def combinationSum_rec(self, candidates: List[int], target: int) -> List[List[int]]:
    output = []
    def backtrack(start, remain, curr):
      if remain == 0: output.append(curr[:])
      if remain < 0: return 0
      for i in range(start, len(candidates)):
        curr.append(candidates[i])
        backtrack(i, remain - candidates[i], curr)
        curr.pop()
    backtrack(0, target, [])
    return output

  def combinationSum_iter(self, candidates: List[int], target: int) -> List[List[int]]:
    output = []
    stack = [(0, target, [])]  # Start with a stack with initial parameters
    while stack:
      start, remain, curr = stack.pop()
      if remain == 0:
        output.append(curr[:])
        continue
      if remain < 0:
        continue
      for i in range(start, len(candidates)):
        # Instead of making a recursive call, add the parameters to the stack
        stack.append((i, remain - candidates[i], curr + [candidates[i]]))
    return output

  def combinationSum(self, candidates, target):
    # Sort candidates to help avoid duplicates
    candidates.sort()
    # dp table where each element is a set of tuples, each tuple is a combination that adds up to the index
    dp = [set() for _ in range(target + 1)]
    dp[0].add(())
    for num in candidates:
      for t in range(num, target + 1):
        for prev in dp[t - num]:
          dp[t].add(prev + (num,))
    # Convert the sets of tuples into a list of lists
    return [list(comb) for comb in dp[target]]

  '''
  40. Combination Sum II
  Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
Each number in candidates may only be used once in the combination.
Note: The solution set must not contain duplicate combinations.Given a collection of candidate numbers (candidates) and a target number (target), find all unique combinations in candidates where the candidate numbers sum to target.
Each number in candidates may only be used once in the combination.
Note: The solution set must not contain duplicate combinations.

Example 1:

Input: candidates = [10,1,2,7,6,1,5], target = 8
Output: 
[
[1,1,6],
[1,2,5],
[1,7],
[2,6]
]
Example 2:

Input: candidates = [2,5,2,1,2], target = 5
Output: 
[
[1,2,2],
[5]
]
  '''
  def combinationSum2(self, candidates: List[int], target: int) -> List[List[int]]:
    output = []
    candidates.sort()
    def backtrack(start, curr, rem):
      if rem == 0: output.append(curr[:])
      for i in range(start, len(candidates)):
        if i > start and candidates[i] == candidates[i - 1]: continue
        if rem - candidates[i] < 0: break  # optimization: skip the rest of elements starting from 'start' index
        curr.append(candidates[i])
        backtrack(i + 1, curr, rem - candidates[i])
        curr.pop()
    backtrack(0, [], target)
    return output
  '''
  Find all valid combinations of k numbers that sum up to n such that the following conditions are true:

Only numbers 1 through 9 are used.
Each number is used at most once.
Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.

Example 1:

Input: k = 3, n = 7
Output: [[1,2,4]]
Explanation:
1 + 2 + 4 = 7
There are no other valid combinations.
Example 2:

Input: k = 3, n = 9
Output: [[1,2,6],[1,3,5],[2,3,4]]
Explanation:
1 + 2 + 6 = 9
1 + 3 + 5 = 9
2 + 3 + 4 = 9
There are no other valid combinations.
Example 3:

Input: k = 4, n = 1
Output: []
Explanation: There are no valid combinations.
Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
  '''
  def combinationSum3_rec(self, k: int, n: int) -> List[List[int]]:
    output = []
    def backtrack(start, curr):
      if len(curr) == k and sum(curr) == n:
        output.append(curr[:])
      if len(curr) > k or sum(curr) > n:
        return
      for i in range(start + 1, 10):
        curr.append(i)
        backtrack(i, curr)
        curr.pop()
    backtrack(0, [])
    return output
  def combinationSum3_iter(self, k: int, n: int) -> List[List[int]]:
      output = []
      stack = [(0, [], 1)]  # (start, current combination, current number to consider)
      while stack:
          start, curr, num = stack.pop()
          if len(curr) == k and sum(curr) == n:
              output.append(curr[:])
              continue
          if len(curr) > k or sum(curr) > n or num > 9: continue
          # Include the current number in the combination
          curr.append(num)
          stack.append((start, curr[:], num + 1))
          # Exclude the current number from the combination
          curr.pop()
          stack.append((num, curr[:], num + 1))
      return output

  '''
  377. Combination Sum IV
  Given an array of distinct integers nums and a target integer target, return the number of possible combinations that add up to target.
The test cases are generated so that the answer can fit in a 32-bit integer.
Example 1:

Input: nums = [1,2,3], target = 4
Output: 7
Explanation:
The possible combination ways are:
(1, 1, 1, 1)
(1, 1, 2)
(1, 2, 1)
(1, 3)
(2, 1, 1)
(2, 2)
(3, 1)
Note that different sequences are counted as different combinations.
Example 2:
Input: nums = [9], target = 3
Output: 0
 Time: O(TN)
Space: O(T)
  '''
  def combinationSum4_iter(self, nums: List[int], target: int) -> int:
    dp = [0] * (target + 1)
    for targ in range(1, target + 1):
      for num in nums:
        diff = targ - num
        if diff == 0:
          dp[targ] += 1
        elif diff > 0:
          dp[targ] += dp[diff]
    return dp[-1]

  '''
  dp = [0, 0, 0, 0, 0]
  targ = 1 i = 0 diff = 0 dp[targ] = 1 dp[diff] = 0  namely, 1 comes fomr 1 or implies 4 comes from 1, 1, 1, 1
  targ = 2 i = 0 diff = 1 dp[targ] = 1 dp[diff] = 1  namely, 2 comes from 1, 1, also implies 4 comes from 1, 1, 1, 1
  targ = 2 i = 1 diff = 0 dp[targ] = 2 dp[diff] = 0  namely, 2 comes from 2, also imples 4 comes from 2, 2
  targ = 3 i = 0 diff = 2 dp[targ] = 2 dp[diff] = 2  3 from 1, 2 and 1, 1, 1,  or 4 from 1, 1, 2 and 1, 1, 1, 1
  targ = 3 i = 1 diff = 1 dp[targ] = 3 dp[diff] = 1
  targ = 3 i = 2 diff = 0 dp[targ] = 4 dp[diff] = 0
  targ = 4 i = 0 diff = 3 dp[targ] = 4 dp[diff] = 4
  targ = 4 i = 1 diff = 2 dp[targ] = 6 dp[diff] = 2
  targ = 4 i = 2 diff = 1 dp[targ] = 7 dp[diff] = 1
  '''
  def combinationSum4_rec(self, nums: List[int], target: int) -> int:
    nums.sort()
    @cache
    def dp(rem):
      if rem == 0: return 1
      res = 0
      for i, num in enumerate(nums):
        if rem - num >= 0:
          res += dp(rem - num)
        else:
          break
      return res
    return dp(target)
  def combinationSum4(self, nums, target):
    @cache
    def backtrack(remaining):
      if remaining == 0: return 1
      if remaining < 0: return 0
      count = 0
      for num in nums:
        count += backtrack(remaining - num)
      return count
    return backtrack(target)

  '''
  90. Subsets II
  Given an integer array nums that may contain duplicates, return all possible 
  subsets (the power set).
  The solution set must not contain duplicate subsets. Return the solution in any order.
  '''
  def subsetsWithDup_dfs(self, nums):
    res = []
    def dfs(start, path):
      res.append(path)
      for i in range(start, len(nums)):
        if i>start and nums[i]==nums[i-1]: continue
        dfs(i+1, path+[nums[i]])
    dfs(0, [])
    return res

  def subsetsWithDup_backtrack(self, nums: List[int]) -> List[List[int]]:
    n, output = len(nums), []
    def backtrack(start, k, curr):
      if len(curr) == k and sorted(curr) not in output:
        output.append(sorted(curr[:]))
      for i in range(start, n):
        curr.append(nums[i])
        backtrack(i + 1, k, curr)
        curr.pop()
    for k in range(n + 1):
      backtrack(0, k, [])
    return output
  def subsetsWithDup_iter(self, nums: List[int]) -> List[List[int]]:
      nums.sort()
      subsetsize, subsets = 0, [[]]
      for i in range(len(nums)):
          start = subsetsize if i>=1 and nums[i]==nums[i-1] else 0
          subsetsize = len(subsets)
          for j in range(start, subsetsize):
              cur_sub = subsets[j][:]
              cur_sub.append(nums[i])
              subsets.append(cur_sub)
      return subsets