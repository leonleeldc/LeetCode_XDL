'''
818. Race Car
Your car starts at position 0 and speed +1 on an infinite number line. Your car can go into negative positions. Your car drives automatically according to a sequence of instructions 'A' (accelerate) and 'R' (reverse):

When you get an instruction 'A', your car does the following:
position += speed
speed *= 2
When you get an instruction 'R', your car does the following:
If your speed is positive then speed = -1
otherwise speed = 1
Your position stays the same.
For example, after commands "AAR", your car goes to positions 0 --> 1 --> 3 --> 3, and your speed goes to 1 --> 2 --> 4 --> -1.

Given a target position target, return the length of the shortest sequence of instructions to get there.
Example 1:

Input: target = 3
Output: 2
Explanation:
The shortest instruction sequence is "AA".
Your position goes from 0 --> 1 --> 3.
Example 2:

Input: target = 6
Output: 5
Explanation:
The shortest instruction sequence is "AAARA".
Your position goes from 0 --> 1 --> 3 --> 7 --> 7 --> 6.
'''
import math
from collections import deque
class RaceCar:
  def racecar_dp(self, target: int) -> int:
    dp = [0] * (target + 1)
    for t in range(1, target + 1):
      n = math.ceil(math.log2(t + 1))
      if 1 << n == t + 1:
        dp[t] = n
      else:
        dp[t] = n + 1 + dp[(1 << n) - 1 - t]
        for m in range(n - 1):
          cur = (1 << (n - 1)) - (1 << m)
          dp[t] = min(dp[t], n + m + 1 + dp[t - cur])
    return dp[target]
  def racecar_rec(self, target: int) -> int:
    @cache
    def dp(t):
      n = math.ceil(math.log2(t + 1))
      if 1 << n == t + 1: return n
      ans = n + 1 + dp((1 << n) - 1 - t)
      for m in range(n - 1):
        cur = (1 << (n - 1)) - (1 << m)
        ans = min(ans, n + m + 1 + dp(t - cur))
      return ans
    return dp(target)
  def racecar_bfs(self, target: int) -> int:
    visited = {(0, 1): True}
    queue = [(0, 1, 0)]  # pos, speed, steps

    def traverse(pos, speed, steps):
      nonlocal queue, visited, target
      if 0 <= pos < 2 * target and (pos, speed) not in visited:
        queue.append([pos, speed, steps + 1])
        visited[(pos, speed)] = True

    while queue:
      pos, speed, steps = queue.pop(0)
      if pos == target: return steps
      traverse(pos + speed, 2 * speed, steps)
      traverse(pos, -1 if speed > 0 else 1, steps)
  def racecar(self, target: int) -> int:
    queue = deque([(0, 1)])
    visited = {(0, 1)}
    steps = 0
    while queue:
      len_q = len(queue)
      while len_q > 0:
        pos, speed = queue.popleft()
        next_pos, next_speed = pos + speed, speed * 2
        if next_pos == target: return steps + 1
        # accelerate
        if abs(next_pos) < 2 * target and abs(next_speed) < 2 * target and (next_pos, next_speed) not in visited:
          queue.append((next_pos, next_speed))
          visited.add((next_pos, next_speed))
        ##reverse
        next_speed2 = -1 if speed > 0 else 1
        if (pos, next_speed2) not in visited:
          queue.append((pos, next_speed2))
          visited.add((pos, next_speed2))
        len_q -= 1
      steps += 1
    return -1

'''
943 Problem
Given an array A of strings, find any smallest string that contains each string in A as a substring.
We may assume that no string in A is substring of another string in A. May you help convert the following cpp to python?
'''
from typing import List
from itertools import permutations
from functools import cache, lru_cache
class FindShortestSuperstring:
  '''
  refers to huahua leetcode
  https://zxi.mytechroad.com/blog/searching/leetcode-943-find-the-shortest-superstring/
  TLE in leetcode
  Try all permutations. Pre-process the cost from word[i] to word[j] and store it in g[i][j].
  Time complexity: O(n!)
  Space complexity: O(n)
  '''
  def shortestSuperstring_search(self, A: List[str]) -> str:
    n = len(A)
    self.g = [[0] * n for _ in range(n)]
    for i in range(n):
      for j in range(n):
        self.g[i][j] = len(A[j])
        for k in range(1, min(len(A[i]), len(A[j])) + 1):
          if A[i][-k:] == A[j][:k]:
            self.g[i][j] = len(A[j]) - k
    path = [0] * n
    self.best_len = float('inf')
    self.dfs(A, 0, 0, 0, path)
    ans = A[self.best_path[0]]
    for k in range(1, len(self.best_path)):
      i, j = self.best_path[k - 1], self.best_path[k]
      ans += A[j][-self.g[i][j]:]
    return ans

  def dfs(self, A, d, used, cur_len, path):
    if cur_len >= self.best_len:
      return
    if d == len(A):
      self.best_len = cur_len
      self.best_path = path.copy()
      return

    for i in range(len(A)):
      if used & (1 << i):
        continue
      path[d] = i
      self.dfs(A, d + 1, used | (1 << i), cur_len + len(A[i]) if d == 0 else cur_len + self.g[path[d - 1]][i], path)

  '''
  Solution 2: DP
g[i][j] is the cost of appending word[j] after word[i], or weight of edge[i][j].
We would like find the shortest path to visit each node from 0 to n – 1 once and only once this is called the Travelling sells man’s problem which is NP-Complete.
We can solve it with DP that uses exponential time.
dp[s][i] := min distance to visit nodes (represented as a binary state s) once and only once and the path ends with node i.
e.g. dp[7][1] is the min distance to visit nodes (0, 1, 2) and ends with node 1, the possible paths could be (0, 2, 1), (2, 0, 1).
Time complexity: O(n^2 * 2^n)
Space complexity: O(n * 2^n)
  '''
  def shortestSuperstring_dp(self, A: List[str]) -> str:
    def connect(w1, w2):
      return [w2[i:] for i in range(len(w1) + 1) if w1[-i:] == w2[:i] or not i][-1]
    N = len(A)
    # Initializing dp with (length, string) values
    dp = [[(float("inf"), "")] * N for _ in range(1 << N)]
    for i in range(N):
      dp[1 << i][i] = (len(A[i]), A[i])
    # Iteratively filling the dp table
    for mask in range(1, 1 << N):  # Starting from 1 to cover all combinations
      n_z_bits = [j for j in range(N) if mask & (1 << j)]
      for j, k in permutations(n_z_bits, 2):
        # This is an optimization. If j is not in the current mask (when subtracting the j-th bit), skip
        if not (mask & (1 << j)):
          continue
        prev_mask = mask ^ (1 << j)
        cand = dp[prev_mask][k][1] + connect(A[k], A[j])
        dp[mask][j] = min(dp[mask][j], (len(cand), cand))
    return min(dp[-1], key=lambda x: x[0])[1]
  def shortestSuperstring_dp_no_permutations(self, A: List[str]) -> str:
    n = len(A)

    # Construct overlap graph
    g = [[0] * n for _ in range(n)]
    for i in range(n):
      for j in range(n):
        g[i][j] = len(A[j])
        for k in range(1, min(len(A[i]), len(A[j])) + 1):
          if A[i][-k:] == A[j][:k]:
            g[i][j] = len(A[j]) - k

    # DP and Parent initialization
    dp = [[float('inf')] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    # Base case for dp: setting the starting lengths
    for i in range(n):
      dp[1 << i][i] = len(A[i])

    # Iterate through all subsets of nodes
    for s in range(1, 1 << n):
      for j in range(n):
        # Continue if j is not in the subset represented by s
        if not (s & (1 << j)):
          continue

        ps = s ^ (1 << j)  # Previous state without j

        for i in range(n):
          if dp[ps][i] + g[i][j] < dp[s][j]:
            dp[s][j] = dp[ps][i] + g[i][j]
            parent[s][j] = i

    # Starting with the last state of DP and backtrack our path.
    mask, j = (1 << n) - 1, dp[-1].index(min(dp[-1]))
    ans = A[j]

    while mask:
      i = parent[mask][j]
      if i >= 0:
        ans = A[i] + ans[len(A[j]) - g[i][j]:]
      mask ^= (1 << j)
      j = i
    return ans
  def shortestSuperstring_mem_rec(self, A):
    '''
    https://leetcode.com/problems/find-the-shortest-superstring/solutions/1225543/python-dp-on-subsets-solution-3-solutions-oneliner-explained/
    '''
    @lru_cache(None)
    def connect(w1, w2):
      return [w2[i:] for i in range(len(w1) + 1) if w1[-i:] == w2[:i] or not i][-1]
    N = len(A)
    dp = [[(float("inf"), "")] * N for _ in range(1 << N)]
    for i in range(N): dp[1 << i][i] = (len(A[i]), A[i])

    for mask in range(1 << N):
      n_z_bits = [j for j in range(N) if mask & 1 << j]

      for j, k in permutations(n_z_bits, 2):
        cand = dp[mask ^ 1 << j][k][1] + connect(A[k], A[j])
        dp[mask][j] = min(dp[mask][j], (len(cand), cand))

    return min(dp[-1])[1]

'''
Problem 698: Partition to K Equal Sum Subsets
Given an integer array nums and an integer k, return true if it is possible to divide this array into k non-empty subsets whose sums are all equal.
thanks to https://leetcode.com/discuss/general-discussion/1125779/dynamic-programming-on-subsets-with-examples-explained
'''
class PartitionKEqualSumSubsets:
  def canPartitionKSubsets_dp1(self, nums, k):
    N = len(nums)
    nums.sort(reverse=True)
    basket, rem = divmod(sum(nums), k)
    if rem or nums[0] > basket: return False
    dp = [-1] * (1 << N)
    dp[0] = 0 ##128 >> 2=32, and 1<<7=128
    for mask in range(1 << N):
      for j in range(N):
        neib = dp[mask ^ (1 << j)] ## mask ^ (1 << j) is used to find all possible combinations of neibor posistions and mask & (1 << j) used to remove impossible positions.
        print(f'mask = {mask} j ={j}  mask ^ (1 << j) = {mask ^ (1 << j)} mask & (1 << j) = {mask & (1 << j)} and neib = {neib}')
        if mask & (1 << j) and neib >= 0 and neib + nums[j] <= basket:
          dp[mask] = (neib + nums[j]) % basket
          print(f'mask & (1 << j) = {mask & (1 << j)} dp[mask] = {dp[mask]}')
          break
    return dp[-1] == 0
  '''
  Note: in the following approach, we use mask & (1 << i))==0 as a right pos and then use  mask | (1 << i) to fill positions.
  somewhat different from dp1 approach
  Time: O(N*2^N)
  Space: O(2^N)
  '''
  def canPartitionKSubsets_dp2(self, nums: List[int], k: int) -> bool:
      total_array_sum, n = sum(nums), len(nums)
      if total_array_sum % k != 0: return False # If the total sum is not divisible by k, we can't make subsets.
      target_sum = total_array_sum // k
      subsetSum = [-1] * (1 << n)
      subsetSum[0] = 0 # Initially only one state is valid, i.e don't pick anything
      for mask in range(1 << n):
          if subsetSum[mask] == -1:  continue # If the current state has not been reached earlier.
          for i in range(n):
              # If the number nums[i] was not picked earlier, and nums[i] + subsetSum[mask]
              # is not greater than the targetSum then add nums[i] to the subset
              # sum at subsetSum[mask] and store the result at subsetSum[mask | (1 << i)].
              if (mask & (1 << i)) == 0 and subsetSum[mask] + nums[i] <= target_sum:
                  subsetSum[mask | (1 << i)] = (subsetSum[mask] + nums[i]) % target_sum
                  print(f'mask & (1 << i) = {mask & (1 << i)}')
          if subsetSum[-1] == 0: return True
      return subsetSum[-1] == 0

  def canPartitionKSubsets_backtrack(self, nums: List[int], k: int) -> bool:
    n, total_sum = len(nums), sum(nums)
    if total_sum % k != 0: return False
    target_sum = total_sum // k
    nums.sort(reverse=True)

    @lru_cache(None)
    def backtrack(index, count, curr_sum, mask):
      if count == k - 1: return True
      if curr_sum > target_sum: return False
      if curr_sum == target_sum:
        return backtrack(0, count + 1, 0, mask)
      for j in range(index, n):
        if ((mask >> j) & 1) == 0:
          new_mask = (mask | (1 << j))
          if backtrack(j + 1, count, curr_sum + nums[j], new_mask): return True
      return False
    return backtrack(0, 0, 0, 0)

'''
473. Matchsticks to Square
You are given an integer array matchsticks where matchsticks[i] is the length of the ith matchstick. You want to use all the matchsticks to make one square. You should not break any stick, but you can link them up, and each matchstick must be used exactly one time.
Return true if you can make this square and false otherwise.
'''
class MatchsticksToSquare:
  def makesquare_dp(self, nums):
    N = len(nums)
    basket, rem = divmod(sum(nums), 4)
    if rem or nums[0] > basket: return False
    dp = [-1] * (1 << N)
    dp[0] = 0
    for mask in range(1 << N):
      for j in range(N):
        neib = dp[mask ^ 1 << j]
        if mask & 1 << j and neib >= 0 and neib + nums[j] <= basket:
          dp[mask] = (neib + nums[j]) % basket
          break
    return dp[-1] == 0

  def makesquare_dfs(self, nums):
    N = len(nums)
    basket, rem = divmod(sum(nums), 4)
    if rem or nums[0] > basket: return False

    @lru_cache(None)
    def dfs(mask):
      if mask == 0: return 0
      for j in range(N):
        if mask & 1 << j:
          neib = dfs(mask ^ 1 << j)
          if neib >= 0 and neib + nums[j] <= basket:
            return (neib + nums[j]) % basket
      return -1
    return dfs((1 << N) - 1) == 0

'''
526. Beautiful Arrangement
Suppose you have n integers labeled 1 through n. A permutation of those n integers perm (1-indexed) is considered a beautiful arrangement if for every i (1 <= i <= n), either of the following is true:
perm[i] is divisible by i.
i is divisible by perm[i].
Given an integer n, return the number of the beautiful arrangements that you can construct.
'''


