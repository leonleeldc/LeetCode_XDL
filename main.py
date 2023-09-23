from DFS import RemoveInvalidParenthesis
from backtrack import WordPattern
wp1 = WordPattern()
pattern = "abab"
str = "redblueredblue"
print(f"solution = {wp1.backtrack(pattern, str, {})}")


# rip = RemoveInvalidParenthesis()
# '''
# Example 1:
#
# Input: s = "()())()"
# Output: ["(())()","()()()"]
# Example 2:
#
# Input: s = "(a)())()"
# Output: ["(a())()","(a)()()"]
# Example 3:
#
# Input: s = ")("
# Output: [""]
# '''
# s = "()())()"
# print(rip.remove_invalid_parentheses(s))