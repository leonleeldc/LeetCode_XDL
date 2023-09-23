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
