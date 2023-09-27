'''
7. Reverse Integer
https://leetcode.com/problems/reverse-integer/
'''
class MathRelated:
  def reverse(self, x: int) -> int:
    if x==0: return x
    flag = 0
    if x < 0:
      flag = 1
      x = abs(x)
    num_arr = []
    while x:
      rem = x % 10
      x = x // 10
      num_arr.append(rem)
    ans = num_arr[0]
    for num in num_arr[1:]:
      ans = ans * 10 + num
    if flag==1:
      ans = -ans
    if ans < -(2 ** 31) or ans >= (2 ** 31):
      return 0
    else:
      return ans
  def reverse_strmethod(self, x: int) -> int:
    flag = 0
    if x < 0:
      flag = 1
      x = abs(x)
    y = str(x)[::-1]
    if flag == 1:
      y = '-' + y
    y = int(y)
    if y < -(2 ** 31) or y >= (2 ** 31):
      return 0
    else:
      return y