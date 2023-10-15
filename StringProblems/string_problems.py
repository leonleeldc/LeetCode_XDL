'''
151. Reverse Words in a String
Example 1:

Input: s = "the sky is blue"
Output: "blue is sky the"
Example 2:

Input: s = "  hello world  "
Output: "world hello"
Explanation: Your reversed string should not contain leading or trailing spaces.
Example 3:

Input: s = "a good   example"
Output: "example good a"
Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
'''
class StringProblemsClass:
  def reverseWords(self, s: str) -> str:
    ss = list(s)
    word = ''
    output = []
    for i in range(len(ss)):
      if ss[i].isalnum():
        word += ss[i]
      if ss[i].isspace() and len(word) > 0:
        output.append(word)
        word = ''
    if len(word) > 0:
      output.append(word)
    ans = ''
    for i in range(len(output) - 1, 0, -1):
      ans += output[i] + ' '
    ans += output[0]
    return ans