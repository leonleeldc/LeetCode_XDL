'''
Given boolean array of days at work, [T, F, T, T, F, F, F, T] and pto (number of PTOs one can take) - where boolean T means paid holiday and F means you can take a PTO. Find the maximum length of vacation an employee can take.
int findMaxVacationLength(year = [F, T, F, T, F, T, F, ], pto = 2) should return 5 because if we take PTO on indices year[2] and year[4], then we can get the maximum length vacation (consecutive T's). May you help write a python script?
'''
def findMaxVacationLength(year, pto):
  max_vacation = 0
  pto_used = 0
  window_start = 0

  for window_end in range(len(year)):
    # If it's a working day, use a PTO (if available)
    if not year[window_end]:
      pto_used += 1

    # If we've used more than the available PTOs, move the window start forward
    while pto_used > pto:
      if not year[window_start]:
        pto_used -= 1
      window_start += 1

    # Update the maximum vacation length
    max_vacation = max(max_vacation, window_end - window_start + 1)
  return max_vacation
'''
301. Remove Invalid Parentheses
Given a string s that contains parentheses and letters, remove the minimum number of invalid parentheses to make the input string valid.
Return a list of unique strings that are valid with the minimum number of removals. You may return the answer in any order.
Example 1:
Input: s = "()())()"
Output: ["(())()","()()()"]
Example 2:
Input: s = "(a)())()"
Output: ["(a())()","(a)()()"]
Example 3:
Input: s = ")("
Output: [""]
'''
from typing import List
from collections import deque
class ParenthesisProblem:
  def removeInvalidParentheses(self, s: str) -> List[str]:
    def is_valid(s):
      cnt = 0
      for ch in s:
        if ch == '(': cnt += 1
        if ch == ')':
          cnt -= 1
          if cnt < 0:  ##this is the key! when cnt<0, even if there are later '(' to balance the number, the loop is done.
            return False
      return cnt == 0
    res, queue, visited = [], deque([(s, 0)]), {s}
    is_found, min_changes = False, len(s)  ##BFS
    while queue:
      cur_s, num_changes = queue.popleft()
      if is_found and min_changes < num_changes: break  ##this break is kind of tricky. This means for later pop out, nums_changes should be always larger than min_num_changes
      if is_valid(cur_s):
        is_found = True
        min_changes = num_changes  ##this condition is important, since we decrease from len(s) to guaranttee that we will not decreas more once a valid one is found.
        res.append(cur_s)
      for i, ch in enumerate(cur_s):
        new_s = cur_s[:i] + cur_s[i + 1:]  # generate all possible strings by removing one char from curr_s
        if new_s not in visited:
          visited.add(new_s)  # we might create duplicate string -> s='())' we remove i=1 or i=2 and result in the same s
          queue.append((new_s, num_changes + 1))
    return res
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
  '''
  424. Longest Repeating Character Replacement
  You are given a string s and an integer k. You can choose any character of the string and change it to any other uppercase English character. You can perform this operation at most k times.
Return the length of the longest substring containing the same letter you can get after performing the above operations.
Example 1:
Input: s = "ABAB", k = 2
Output: 4
Explanation: Replace the two 'A's with two 'B's or vice versa.
Example 2:
Input: s = "AABABBA", k = 1
Output: 4
Explanation: Replace the one 'A' in the middle with 'B' and form "AABBBBA".
The substring "BBBB" has the longest repeating letters, which is 4.
There may exists other ways to achieve this answer too
  '''
  def characterReplacement(self, s: str, k: int) -> int:
    max_len, cur = 0, 1
    i = 1
    for ch in s[1:]:
      if ch == s[i - 1]:
        cur += 1
      else:
        cur += k if i + k <= len(s) else len(s) - i + 1
        max_len = max(max_len, cur)
        cur = 1
      i += 1
    max_len = max(max_len, cur)
    return max_len

  '''
  counter(s), obtain c: cnt
  prob1:
  k=2
  A:2
  B:2
  since len(s)==4, we can make sure, the output is 4
  prob2:
  k=1
  AABABBA
  A:4
  B:3
  find how many ch are connected,
  '''
  def characterReplacement_slide_window(self, s: str, k: int) -> int:
    max_len = 0
    max_count = 0
    count = {}

    left = 0
    for right in range(len(s)):
      count[s[right]] = count.get(s[right], 0) + 1
      max_count = max(max_count, count[s[right]])

      # If the number of chars to be replaced is more than k, shrink the window
      while right - left + 1 - max_count > k:
        count[s[left]] -= 1
        left += 1

      # Update max_len if the current window is larger
      max_len = max(max_len, right - left + 1)

    return max_len
