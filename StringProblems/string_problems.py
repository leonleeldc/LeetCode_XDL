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
  '''
  5. Longest Palindromic Substring
  '''
  def longestPalindrome_TLE(self, s: str) -> str:
      max_len, start, end = 0, 0, 0
      def is_palindrome(i, j):
          while i<j:
              if s[i]!=s[j]: return False
              i+=1
              j-=1
          return True
      for i in range(len(s)):
          for j in range(i, len(s)):
              if is_palindrome(i, j) and j-i>max_len:
                  max_len = j-i
                  start, end = i, j
      return s[start:end+1]

  def longestPalindrome_dp(self, s: str) -> str:
    def expand_from_mid(s, i, j):
      if (s == None or len(s) == 0): return ''
      while i >= 0 and j < len(s) and s[i] == s[j]:
        i -= 1
        j += 1
      return j - i - 1

    leng, start = 0, 0
    for i in range(len(s)):
      curr = max(expand_from_mid(s, i, i), expand_from_mid(s, i, i + 1))
      if curr <= leng: continue
      leng = curr
      start = i - (curr - 1) // 2
    return s[start:start + leng]

  def customSort(self, input_str, order):
    # 创建一个哈希表存储 order 中字符的索引
    order_index = {char: idx for idx, char in enumerate(order)}
    # 对 input_str 中的字符进行排序
    sorted_str = sorted(input_str, key=lambda x: order_index.get(x, len(order)))
    # 返回排序后的字符串
    return ''.join(sorted_str)

  def customSort_followup(self, input_str, order):
    order_index = {char: idx for idx, char in enumerate(order)}
    # 使用 order 中的索引排序，如果字符不在 order 中，则赋予一个较大的索引值来将其放在末尾
    sorted_str = sorted(input_str, key=lambda x: order_index.get(x, len(order) + 1))
    return ''.join(sorted_str)

