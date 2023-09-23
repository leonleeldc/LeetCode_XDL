from DFS import RemoveInvalidParenthesis, WordPattern
wp1 = WordPattern()
pattern = "abab"
str = "redblueredblue"
print(f"solution = {wp1.dfs(pattern, str, {})}")


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