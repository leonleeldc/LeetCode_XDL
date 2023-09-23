# 780 · Remove Invalid Parentheses
# Description
# Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.
from typing import (
    List,
)

class RemoveInvalidParenthesis:
    """
    @param s: The input string
    @return: Return all possible results
             we will sort your return value in output
    """
    def remove_invalid_parentheses(self, s: str) -> List[str]:
        # Write your code here
        def isValid(s):
            count = 0
            for ch in s:
                if ch == '(':
                    count += 1
                elif ch == ')':
                    count -= 1
                if count < 0:
                    return False
            return count == 0
        
        def dfs(s, start, l, r, ans):
            if l == 0 and r == 0:
                if isValid(s):
                    ans.append(s)
                return
            
            for i in range(start, len(s)):
                if i != start and s[i] == s[i - 1]:
                    continue
                
                if s[i] == '(' or s[i] == ')':
                    curr = s[:i] + s[i + 1:]
                    if r > 0 and s[i] == ')':
                        dfs(curr, i, l, r - 1, ans)
                    elif l > 0 and s[i] == '(':
                        dfs(curr, i, l - 1, r, ans)
            
        l = 0
        r = 0

        for ch in s:
            l += (ch == '(')
            if l == 0:
                r += (ch == ')')
            else:
                l -= (ch == ')')
        
        ans = []
        dfs(s, 0, l, r, ans)
        return ans    

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
        return self.dfs(pattern, string, {})
    def dfs(self, ptn, s, map):
        if not ptn: return not s
        if ptn[0] in map:
            prefix = map[ptn[0]]
            if not s.startswith(prefix): return False
            return self.dfs(ptn[1:], s[len(prefix):], map)
        for i in range(1, len(s) + 1):
            prefix = s[:i]
            if prefix in map.values(): continue
            map[ptn[0]] = prefix
            if self.dfs(ptn[1:], s[len(prefix):], map): return True
            del map[ptn[0]]
        return False

