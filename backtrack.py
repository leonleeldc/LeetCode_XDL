'''
22. Generate Parentheses
'''
from typing import List
class GenParentheses:
    def generateParenthesis_rec(self, n: int) -> List[str]:
        output = []
        def backtrack(i, j, curr):
            if len(curr)==2*n:
                output.append(''.join(curr))
                return
            if i<n:
                curr.append('(')
                backtrack(i+1, j, curr)
                curr.pop()
            if j<i:
                curr.append(')')
                backtrack(i, j+1, curr)
                curr.pop()
        backtrack(0, 0, [])
        return output
    def generateParenthesis_iter(self, n: int) -> List[str]:
        dp = [[] for _ in range(n+1)]
        dp[0].append('')
        for i in range(1, n+1):
            for j in range(i):
                dp[i] += ['(' + x + ')' + y for x in dp[j] for y in dp[i-j-1]]
        return dp[n]
# 829 · Word Pattern II
# Description
# Given a pattern and a string str, find if str follows the same pattern.
'''
Example 1

Input:
pattern = "abab"
str = "redblueredblue"
Output: true
Explanation: "a"->"red","b"->"blue"
Example 2

Input:
pattern = "aaaa"
str = "asdasdasdasd"
Output: true
Explanation: "a"->"asd"
Example 3

Input:
pattern = "aabb"
str = "xyzabcxzyabc"
Output: false
'''
# Here follow means a full match, such that there is a bijection between a letter in pattern and a non-empty substring in str.(i.e if a corresponds to s, then b cannot correspond to s. For example, given pattern = "ab", str = "ss", return false.)
# Solution 1
class WordPattern:
    def wordPatternMatch(self, pattern, string):
        return self.backtrack(pattern, string, {})
    def backtrack(self, ptn, s, map):
        if not ptn: return not s
        if ptn[0] in map:
            prefix = map[ptn[0]]
            if not s.startswith(prefix): return False
            return self.backtrack(ptn[1:], s[len(prefix):], map)
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            if prefix in map.values(): continue
            map[ptn[0]] = prefix
            if self.backtrack(ptn[1:], s[len(prefix):], map): return True
            del map[ptn[0]]
        return False
class DiverseBackTrack:
    def print_products(self, primes, index=0, current_product=1):
        # Base condition: if we have processed all primes
        if index == len(primes):
            # Only print if the current_product is not 1 (we don't consider the case where no primes are chosen)
            if current_product != 1:
                print(current_product)
            return
        # Exclude the current prime and move on to the next
        self.print_products(primes, index + 1, current_product)
        # Include the current prime in the product and move on to the next
        self.print_products(primes, index + 1, current_product * primes[index])
'''
 Print all possible products of the given primes
where each prime can be used either 0 or 1 times.
primes = [2, 3,11]
Want to print (any order): 2, 3, 11, 6, 33, 22, 66
'''
class ProductOfPrimes:
    def __init__(self):
        self.rets = set()

    def get_product(self, arr):
        for i in range(len(arr)):
            self.dfs(i, arr, 1)
        return self.rets

    def dfs(self, pos, arr, curprod):
        if pos == len(arr):
            return

        nextprod = curprod * arr[pos]
        self.rets.add(nextprod)

        for i in range(pos + 1, len(arr)):
            self.dfs(i, arr, nextprod)


