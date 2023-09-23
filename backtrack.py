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