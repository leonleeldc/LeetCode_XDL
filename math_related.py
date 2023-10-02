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
  def myPow(self, x: float, n: int) -> float:
    if n<0:
      x = 1/x
      n = n*(-1)
    i = 0
    carry = 1
    while n>1:
      if n%2 == 0:
        x = x*x
        n = n/2
      else:
        carry = carry*x
        n = n-1
        x = x*x
        n = n/2
    res = x*carry
    return res if n!=0 else 1
  def myPow_rec(self, x:float, n:int)->float:
    if x < -2 ** 31 or x > 2 ** 31 - 1:
      return

    negFlag = False
    if n < 0:
      negFlag = True
      n = -n

    result = self.pow(x, n)

    if negFlag:
      result = 1 / result

    return result

  def pow(self, x: float, n: int):
    if n == 0:
      return 1

    result = x
    times = 1
    while times < n:
      if times * 2 <= n:
        result = result * result
        times *= 2
      else:
        result = result * pow(x, n - times)
        break
    return result
