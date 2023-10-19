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
# import nltk
# from nltk.corpus import stopwords
# nltk.download('stopwords')
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
  #method to generate n-grams:
  #params:
  #text-the text for which we have to generate n-grams
  #ngram-number of grams to be generated from the text(1,2,3,4 etc., default value=1)
  def generate_N_grams(self, text,ngram=1):
    words=[word for word in text.split(" ") if word not in set(stopwords.words('english'))]
    print("Sentence after removing stopwords:",words)
    temp=zip(*[words[i:] for i in range(0,ngram)])
    ans=[' '.join(ngram) for ngram in temp]
    return ans
  '''
  43. Multiply Strings
  Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2, also represented as a string.
Note: You must not use any built-in BigInteger library or convert the inputs to integer directly.
Example 1:
Input: num1 = "2", num2 = "3"
Output: "6"
Example 2:
Input: num1 = "123", num2 = "456"
Output: "56088
  '''
  def multiply(self, num1: str, num2: str) -> str:
    if '0' in [num1, num2]: return '0'
    num1, num2 = num1[::-1], num2[::-1]
    m, n = len(num1), len(num2)
    res = [0] * (m + n)
    for i in range(m):
      for j in range(n):
        res[i + j] += int(num1[i]) * int(num2[j])
        res[i + j + 1] += res[i + j] // 10
        res[i + j] = res[i + j] % 10
    res, beg = res[::-1], 0
    while beg < len(res) and res[beg] == 0: #aims at removing beginning zeros
      beg += 1
    return ''.join(map(str, res[beg:]))
