from typing import List
from collections import defaultdict
from collections import Counter
import itertools
from functools import lru_cache
import math
'''
1258. Synonymous Sentences
You are given a list of equivalent string pairs synonyms where synonyms[i] = [si, ti] indicates that si and ti are equivalent strings. You are also given a sentence text.
Return all possible synonymous sentences sorted lexicographically.
Example 1:

Input: synonyms = [["happy","joy"],["sad","sorrow"],["joy","cheerful"]], text = "I am happy today but was sad yesterday"
Output: ["I am cheerful today but was sad yesterday","I am cheerful today but was sorrow yesterday","I am happy today but was sad yesterday","I am happy today but was sorrow yesterday","I am joy today but was sad yesterday","I am joy today but was sorrow yesterday"]
Example 2:

Input: synonyms = [["happy","joy"],["cheerful","glad"]], text = "I am happy today but was sad yesterday"
Output: ["I am happy today but was sad yesterday","I am joy today but was sad yesterday"]
'''
from typing import List
from collections import defaultdict
class SynonymousSentences:
    def generateSentences(self, synonyms: List[List[str]], text: str) -> List[str]:
        adj_graph = defaultdict(list)
        # Build the adjacency graph
        for a, b in synonyms:
            adj_graph[a].append(b)
            adj_graph[b].append(a)
        words = text.split()
        included = {word for word in words if word in adj_graph}
        output = set()

        def get_synonyms(word: str) -> List[str]:
            stack = [word]
            synonyms_set = set()
            while stack:
                w = stack.pop()
                if w not in synonyms_set:
                    synonyms_set.add(w)
                    for neighbor in adj_graph[w]:
                        stack.append(neighbor)
            return sorted(synonyms_set)

        def backtrack(index: int, current_sentence: List[str]):
            if index == len(words):
                output.add(' '.join(current_sentence))
                return

            current_word = words[index]
            if current_word in included:
                for synonym in get_synonyms(current_word):
                    current_sentence[index] = synonym
                    backtrack(index + 1, current_sentence)
            else:
                backtrack(index + 1, current_sentence)

        backtrack(0, words[:])
        return sorted(output)

class SynonymousSentencesV2:
    def generateSentences(self, synonyms: List[List[str]], text: str) -> List[str]:
        adj_graph = defaultdict(list)
        # Build the adjacency graph
        for a, b in synonyms:
            adj_graph[a].append(b)
            adj_graph[b].append(a)
        words = text.split()
        included = {word for word in words if word in adj_graph}
        output = set()

        def get_synonyms(word: str, synonyms_set: set[str]) -> List[str]:
            synonyms_set.add(word)
            for next_word in adj_graph[word]:
                if next_word not in synonyms_set:
                    get_synonyms(next_word, synonyms_set)
            return synonyms_set

        def backtrack(index: int, current_sentence: List[str]):
            stack = [(index, current_sentence)]
            while stack:
                index, current_sentence = stack.pop()
                if index == len(words):
                    output.add(' '.join(current_sentence))
                    continue
                current_word = words[index]
                if current_word in included:
                    for synonym in sorted(get_synonyms(current_word, set())):
                        new_sentence = current_sentence[:]
                        new_sentence[index] = synonym
                        stack.append((index+1, new_sentence))
                else:
                    stack.append((index+1, current_sentence))

        backtrack(0, words[:])
        return sorted(output)

'''
1307. Verbal Arithmetic Puzzle
Solved
Hard
Topics
Companies
Hint
Given an equation, represented by words on the left side and the result on the right side.

You need to check if the equation is solvable under the following rules:

Each character is decoded as one digit (0 - 9).
No two characters can map to the same digit.
Each words[i] and result are decoded as one number without leading zeros.
Sum of numbers on the left side (words) will equal to the number on the right side (result).
Return true if the equation is solvable, otherwise return false.

from low digit position to high digit position, stop searching once (left side sum)%10 did not equal right side at current digit
9567 + 1085 = 10652
SEND   MORE   MONEY 

"SIX","SEVEN","SEVEN"  = "TWENTY"
 650   68782   68782      138214
'''


class VerbalArithmeticPuzzle:
    def isSolvable(self, words: List[str], result: str) -> bool:
        if max(map(len, words)) > len(result): return False  # edge case

        words.append(result)
        digits = [0] * 10
        mp = {}  # mapping from letter to digit

        def fn(i, j, val):
            """Find proper mapping for words[i][~j] and result[~j] via backtracking."""
            if j == len(result): return val == 0  # base condition
            if i == len(words): return val % 10 == 0 and fn(0, j + 1, val // 10)

            if j >= len(words[i]): return fn(i + 1, j, val)
            if words[i][~j] in mp:
                if j and j + 1 == len(words[i]) and mp[words[i][~j]] == 0: return  # backtrack (no leading 0)
                if i + 1 == len(words):
                    return fn(i + 1, j, val - mp[words[i][~j]])
                else:
                    return fn(i + 1, j, val + mp[words[i][~j]])
            else:
                for k, x in enumerate(digits):
                    if not x and (k or j == 0 or j + 1 < len(words[i])):
                        mp[words[i][~j]] = k
                        digits[k] = 1
                        if i + 1 == len(words) and fn(i + 1, j, val - k): return True
                        if i + 1 < len(words) and fn(i + 1, j, val + k): return True
                        digits[k] = 0
                        mp.pop(words[i][~j])

        return fn(0, 0, 0)

def addOperators(num, target):
    def backtrack(index, prev_operand, current_operand, value, string):
        # If we have reached the end of the string
        if index == len(num):
            # If the current value is equal to the target and
            # the current operand is 0 (no unfinished operand at the end)
            if value == target and current_operand == 0:
                # Append the expression to the result
                results.append("".join(string))
            return

        # Extending the current operand by one digit
        current_operand = current_operand * 10 + int(num[index])
        str_op = str(current_operand)

        # If the current operand is not 0 (no leading zero allowed except for 0 itself)
        if current_operand > 0:
            # Continue the expression without adding an operator
            backtrack(index + 1, prev_operand, current_operand, value, string)

        # Addition
        string.append('+')
        string.append(str_op)
        backtrack(index + 1, current_operand, 0, value + current_operand, string)
        string.pop()  # backtrack
        string.pop()

        # Can't subtract before having a number, skip if index is 0 (start of string)
        if index > 0:
            # Subtraction
            string.append('-')
            string.append(str_op)
            backtrack(index + 1, -current_operand, 0, value - current_operand, string)
            string.pop()  # backtrack
            string.pop()

    results = []
    backtrack(0, 0, 0, 0, [])
    return results

# Example usage:
num = "123"
target = 6
print(addOperators(num, target))

'''
465. Optimal Account Balancing
You are given an array of transactions transactions where transactions[i] = [fromi, toi, amounti] indicates that the person with ID = fromi gave amounti $ to the person with ID = toi.
Return the minimum number of transactions required to settle the debt.
Example 1:

Input: transactions = [[0,1,10],[2,0,5]]
Output: 2
Explanation:
Person #0 gave person #1 $10.
Person #2 gave person #0 $5.
Two transactions are needed. One way to settle the debt is person #1 pays person #0 and #2 $5 each.
Example 2:

Input: transactions = [[0,1,10],[1,0,1],[1,2,5],[2,0,5]]
Output: 1
Explanation:
Person #0 gave person #1 $10.
Person #1 gave person #0 $1.
Person #1 gave person #2 $5.
Person #2 gave person #0 $5.
Therefore, person #1 only need to give person #0 $4, and all debt is settled.
 

Constraints:

1 <= transactions.length <= 8
transactions[i].length == 3
0 <= fromi, toi < 12
fromi != toi
1 <= amounti <= 100
'''
class OptimalAccountBalancing:
    def minTransfers(self, transactions: List[List[int]]) -> int:
        def dfs(start):
            r"""
            Recursive function will return the minimun number of transactions needed
			to make all([a == 0 for a in accounts[start:]])
            """
            # If your start index reach to end, return 0
            if start == len(accounts): return 0
            cur_balance = accounts[start]
            # If your start index is cleared, clear following debts
            if cur_balance == 0: return dfs(start + 1)
            min_trans = float('inf')
            # Going through following account amount,
            # to see if you can use the combinations
            # to get the minimun number of transactions to cleared accounts[start:]
            for i in range(start + 1, len(accounts)):
                next_balance = accounts[i]
                # The current balance can be reduced only when next balance and current balance are different sign
                if (cur_balance * next_balance) < 0:
                    # accounts[start] makes a transaction to accounts[i] => 1 transaction
                    accounts[i] += cur_balance
                    # move to next position to cleared accounts[(start + 1):]
                    min_trans = min(min_trans, 1 + dfs(start + 1))
                    # recovered to try with clearing this with other position
                    accounts[i] -= cur_balance
                    # A way to prune:
                    # when your accounts[start] == accounts[i], transaction between start and i should be the best case;
                    # therefore, no need to look for following combinations
                    if cur_balance + next_balance == 0: break
            return min_trans
        # balance[k] = the amount of debt k currently has
        balance = Counter()
        for f, t, a in transactions:
            balance[f] -= a
            balance[t] += a
        accounts = [val for a, val in balance.items() if val != 0]
        return dfs(0)
    def minTransfers_optimized(self, transactions: List[List[int]]) -> int:
        tuplify = lambda balance: tuple(sorted((k, v) for k, v in balance.items()))
        @lru_cache(None)
        def dfs(balances):
            if not balances:
                return 0
            res = math.inf
            balances = {k: v for k, v in balances}
            for size in range(2, len(balances) + 1):
                for group in itertools.combinations(balances.keys(), size):
                    if sum(balances[k] for k in group) == 0:
                        remaining_balances = {k: v for k, v in balances.items() if k not in group}
                        res = min(res, size - 1 + dfs(tuplify(remaining_balances)))
            return res
        balances = defaultdict(int)
        for u, v, z in transactions:
            balances[u] += z
            balances[v] -= z
        return dfs(tuplify({k: v for k, v in balances.items() if v}))
    def minTransfers_bitmap(self, transactions: List[List[int]]) -> int:
        # key = id, val = incoming (neg when outcoming)
        bal_dict = defaultdict(int)
        for a, b, amount in transactions:
            bal_dict[a] -= amount
            bal_dict[b] += amount
        balance = [val for a, val in bal_dict.items() if val != 0]
        # find max_set in balance by bitmap
        n = len(balance)
        if n == 0:
            return 0
        tot = [0] * (1 << n)
        max_set = [0] * (1 << n)
        # init state, no one is selected, then select one more
        # next, based on pre selection, select one more
        for select in range(1 << n):
            for shift in range(n):
                more_select = select | (1 << shift)
                if more_select == select:
                    continue
                # else:
                tot[more_select] = tot[select] + balance[shift]
                if tot[more_select] == 0:
                    max_set[more_select] = max(max_set[more_select],
                                                max_set[select] + 1)
                else:
                    max_set[more_select] = max(max_set[more_select],
                                                max_set[select])
        return n - max_set[-1]
'''
22. Generate Parentheses
'''

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


