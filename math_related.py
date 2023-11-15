'''
229. Majority Element II
Given an integer array of size n, find all elements that appear more than ⌊ n/3 ⌋ times.
Example 1:
Input: nums = [3,2,3]
Output: [3]
Example 2:

Input: nums = [1]
Output: [1]
Example 3:

Input: nums = [1,2]
Output: [1,2]
'''
from typing import List

'''
334. Increasing Triplet Subsequence
Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.

 

Example 1:

Input: nums = [1,2,3,4,5]
Output: true
Explanation: Any triplet where i < j < k is valid.
Example 2:

Input: nums = [5,4,3,2,1]
Output: false
Explanation: No triplet exists.
Example 3:

Input: nums = [2,1,5,0,4,6]
Output: true
Explanation: The triplet (3, 4, 5) is valid because nums[3] == 0 < nums[4] == 4 < nums[5] == 6.

'''
class IncreasingTripletSubsequence:
  def increasingTriplet(self, nums: List[int]) -> bool:
    '''
    [1, 90, 2, 1, 0, 2, 3]
    looks quite simliar to longest subsequence
    1 < first, first = 1
    90<second, second =90
    2 < second, second = 2
    1 == first, first=1
    0 < first, first =0
    2 ==second, second = 2
    3 is in else, return True
    '''
    first = second = float('inf')
    for n in nums:
      if n <= first:
        first = n
      elif n <= second:
        second = n
      else:
        return True
    return False

def count_trailing_zeros(n):
  count = 0
  i = 5
  while (n // i) > 0:
    count += n // i
    i *= 5
  return count

def find_pythagorean_triplets(arr):
  squared_arr = [x ** 2 for x in arr]
  squared_arr.sort()
  n = len(squared_arr)
  for i in range(n - 1, 1, -1):
    c = squared_arr[i]
    left = 0
    right = i - 1
    while left < right:
      if squared_arr[left] + squared_arr[right] == c:
        return True
      elif squared_arr[left] + squared_arr[right] < c:
        left += 1
      else:
        right -= 1
  return False


class MajorityElement:
  '''
  Boyer-Moore Voting Algorithm
  '''
  def majorityElementII(self, nums: List[int]) -> List[int]:
    if not nums: return []
    # Step 1: Find potential candidates for majority element
    count1, count2, candidate1, candidate2 = 0, 0, 0, 1
    for num in nums:
      if num == candidate1:
        count1 += 1
      elif num == candidate2:
        count2 += 1
      elif count1 == 0:
        candidate1, count1 = num, 1
      elif count2 == 0:
        candidate2, count2 = num, 1
      else:
        count1, count2 = count1 - 1, count2 - 1
    # Step 2: Verify the candidates
    result = []
    for candidate in [candidate1, candidate2]:
      if nums.count(candidate) > len(nums) // 3:
        result.append(candidate)
    return result
  '''
  169. Majority Element
  Given an array nums of size n, return the majority element.
  The majority element is the element that appears more than ⌊n / 2⌋ times. You may assume that the majority element always exists in the array.
  '''
  def majorityElementI(self, nums: List[int])->int:
    pick, count = 0, 0
    for num in nums:
      pick = (pick, num)[not count]
      count += (-1, 1)[num==pick]
    return pick
  '''
  Boyer-Moore Voting Algorithm
  '''
  def majorityElementI_BMVA(self, nums: List[int])->int:
    cnt, cand = 1, nums[0]
    for i in range(1, len(nums)):
      if nums[i]==cand:
        cnt += 1
      elif nums[i]!=cand and cnt>0:
        cnt -= 1
      elif nums[i]!=cand and cnt==0:
        cnt += 1
        cand = nums[i]
    return cand

'''
65. Valid Number
A valid number can be split up into these components (in order):

A decimal number or an integer.
(Optional) An 'e' or 'E', followed by an integer.
A decimal number can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One of the following formats:
One or more digits, followed by a dot '.'.
One or more digits, followed by a dot '.', followed by one or more digits.
A dot '.', followed by one or more digits.
An integer can be split up into these components (in order):

(Optional) A sign character (either '+' or '-').
One or more digits.
For example, all the following are valid numbers: ["2", "0089", "-0.1", "+3.14", "4.", "-.9", "2e10", "-90E3", "3e+7", "+6e-1", "53.5e93", "-123.456e789"], while the following are not valid numbers: ["abc", "1a", "1e", "e3", "99e2.5", "--6", "-+3", "95a54e53"].

Given a string s, return true if s is a valid number.
'''
import re
class ValidNumber(object):
  def isNumber_re(self, s: str) -> bool:
    pattern = r'^\s*[+-]?(\d+\.?|\.\d)\d*(E[+-]?\d+)?(e[+-]?\d+)?\s*$'
    return re.match(pattern, s)
  class Solution:
    def isNumber(self, s: str) -> bool:
      # Define the regular expression for a valid number
      pattern = re.compile(r"""
              ^                      # start of string
              [+-]?                  # optional sign
              (                      # start of group for the main number part
              (\d+\.\d*|\.\d+)     # decimal number with at least one digit
              |                    # ...or...
              \d+                  # integer with at least one digit
              )                      # end of group for the main number part
              ([eE][+-]?\d+)?        # optional exponent part
              $                      # end of string
              """, re.VERBOSE)

      # Use the pattern to match the input string
      return re.match(pattern, s) is not None
  def isNumber(self, s):
    # This is the DFA we have designed above
    dfa = [
      {"digit": 1, "sign": 2, "dot": 3},
      {"digit": 1, "dot": 4, "exponent": 5},
      {"digit": 1, "dot": 3},
      {"digit": 4},
      {"digit": 4, "exponent": 5},
      {"sign": 6, "digit": 7},
      {"digit": 7},
      {"digit": 7}
    ]

    current_state = 0
    for c in s:
      if c.isdigit():
        group = "digit"
      elif c in ["+", "-"]:
        group = "sign"
      elif c in ["e", "E"]:
        group = "exponent"
      elif c == ".":
        group = "dot"
      else:
        return False

      if group not in dfa[current_state]:
        return False

      current_state = dfa[current_state][group]

    return current_state in [1, 4, 7]

  '''
  Time: O(N)
  Space: O(1)
  '''

'''
681. Next Closest Time
Given a time represented in the format "HH:MM", form the next closest time by reusing the current digits. There is no limit on how many times a digit can be reused.
You may assume the given input string is always valid. For example, "01:34", "12:09" are all valid. "1:34", "12:9" are all invalid.
Example 1:
Input: time = "19:34"
Output: "19:39"
Explanation: The next closest time choosing from digits 1, 9, 3, 4, is 19:39, which occurs 5 minutes later.
It is not 19:33, because this occurs 23 hours and 59 minutes later.
Example 2:
Input: time = "23:59"
Output: "22:22"
Explanation: The next closest time choosing from digits 2, 3, 5, 9, is 22:22.
It may be assumed that the returned time is next day's time since it is smaller than the input time numerically.
'''
import heapq
'''
670. Maximum Swap
You are given an integer num. You can swap two digits at most once to get the maximum valued number.
Return the maximum valued number you can get.
'''

class MaximumSwap:
  def maximumSwap_numerical(self, num: int) -> int:
    hd = hp = ld = lp = 0
    cur_hd, cur_hp, pos, res = -1, 0, 1, num
    while num:
      digit = num % 10
      if digit > cur_hd:
        cur_hd, cur_hp = digit, pos
      elif digit < cur_hd:
        hd, hp = cur_hd, cur_hp,
        ld, lp = digit, pos
      num //= 10
      pos *= 10
      # print(f'num = {num} pos = {pos} lp={lp} hp={hp} ld={ld} hd={hd} digit = {digit}')
    res -= (hd - ld) * (hp - lp)
    # print(res)
    return res
  def maximumSwap_str(self, nums: int) -> int:
    m = nums
    n = str(nums)
    for i in range(len(n)):
      num = list(n)
      for j in range(len(n)):
        if num[j] != num[i]:
          num[i], num[j] = num[j], num[i]
          m = max(m, int(''.join(num)))
          num = list(n)  # we used again convert to orignal list
    return m

  '''
  num = 273 pos = 10 lp=0 hp=0 ld=0 hd=0 digit = 6
  num = 27 pos = 100 lp=10 hp=1 ld=3 hd=6 digit = 3
  num = 2 pos = 1000 lp=10 hp=1 ld=3 hd=6 digit = 7
  num = 0 pos = 10000 lp=1000 hp=100 ld=2 hd=7 digit = 2
  7236
  7236-2736=4500
  '''
class NextClosestTime:
  def nextClosestTime(self, time):
    hours, minutes = time.split(":")
    digits = {x for x in time if x != ":"}
    if len(digits) == 1: return time

    def is_valid(h, m):
      return (0 <= h < 24) and (0 <= m < 60)

    cur_minutes = int(hours) * 60 + int(minutes)
    result_time = None
    min_diff = float('inf')

    for h1 in digits:
      for h2 in digits:
        for m1 in digits:
          for m2 in digits:
            h, m = int(h1 + h2), int(m1 + m2)
            if is_valid(h, m):
              new_minutes = h * 60 + m
              diff = (new_minutes - cur_minutes) % (24 * 60)
              if 0 < diff < min_diff:
                result_time = f"{h1}{h2}:{m1}{m2}"
                min_diff = diff

    return result_time
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

  import heapq

  def merge_iterators(self, *iterators):
    # Convert iterators to lists for convenience
    iterators = [list(it) for it in iterators]

    # Create a min heap and push the first element of each iterator
    min_heap = [(lst[0], i, 0) for i, lst in enumerate(iterators) if lst]
    heapq.heapify(min_heap)

    while min_heap:
      val, list_idx, element_idx = heapq.heappop(min_heap)
      yield val  # Produce merged output one by one

      if element_idx + 1 < len(iterators[list_idx]):
        next_val = iterators[list_idx][element_idx + 1]
        heapq.heappush(min_heap, (next_val, list_idx, element_idx + 1))

  # Test the function
  it1 = iter([1, 4, 7])
  it2 = iter([2, 5, 8])
  it3 = iter([3, 6, 9])

  merged = list(merge_iterators(it1, it2, it3))
  print(merged)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

  '''
  973. K Closest Points to Origin
  Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).


  '''
from typing import List
import math
'''
346. Moving Average from Data Stream
Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

Implement the MovingAverage class:

MovingAverage(int size) Initializes the object with the size of the window size.
double next(int val) Returns the moving average of the last size values of the stream.
 

Example 1:

Input
["MovingAverage", "next", "next", "next", "next"]
[[3], [1], [10], [3], [5]]
Output
[null, 1.0, 5.5, 4.66667, 6.0]

Explanation
MovingAverage movingAverage = new MovingAverage(3);
movingAverage.next(1); // return 1.0 = 1 / 1
movingAverage.next(10); // return 5.5 = (1 + 10) / 2
movingAverage.next(3); // return 4.66667 = (1 + 10 + 3) / 3
movingAverage.next(5); // return 6.0 = (10 + 3 + 5) / 3
'''
from collections import deque
class MovingAverage:
    def __init__(self, size: int):
        self.size = size
        self.queue = deque()
        # number of elements seen so far
        self.window_sum = 0
        self.count = 0

    def next(self, val: int) -> float:
        self.count += 1
        # calculate the new sum by shifting the window
        self.queue.append(val)
        tail = self.queue.popleft() if self.count > self.size else 0
        self.window_sum = self.window_sum - tail + val
        return self.window_sum / min(self.size, self.count)

class KClosestToOrigin:
  def kClosest(self, points: List[List[int]], k: int) -> List[List[int]]:
    # Precompute the Euclidean distance for each point
    distances = [self.euclidean_distance(point) for point in points]
    # Create a reference list of point indices
    remaining = [i for i in range(len(points))]
    # Define the initial binary search range
    low, high = 0, max(distances)

    # Perform a binary search of the distances
    # to find the k closest points
    closest = []
    while k:
      mid = (low + high) / 2
      closer, farther = self.split_distances(remaining, distances, mid)
      if len(closer) > k:
        # If more than k points are in the closer distances
        # then discard the farther points and continue
        remaining = closer
        high = mid
      else:
        # Add the closer points to the answer array and keep
        # searching the farther distances for the remaining points
        k -= len(closer)
        closest.extend(closer)
        remaining = farther
        low = mid

    # Return the k closest points using the reference indices
    return [points[i] for i in closest]

  def split_distances(self, remaining: List[int], distances: List[float],
                      mid: int) -> List[List[int]]:
    """Split the distances around the midpoint
    and return them in separate lists."""
    closer, farther = [], []
    for index in remaining:
      if distances[index] <= mid:
        closer.append(index)
      else:
        farther.append(index)
    return [closer, farther]

  def euclidean_distance(self, point: List[int]) -> float:
    """Calculate and return the squared Euclidean distance."""
    return point[0] ** 2 + point[1] ** 2
  def kClosest_sorting(self, points: List[List[int]], k: int) -> List[List[int]]:
    import math
    distances, res_list = [], []
    for point in points:
      distances.append(math.sqrt(point[0] ** 2 + point[1] ** 2))
    sorted_points = sorted(zip(points, distances), key=lambda x: x[1])
    for i in range(k):
      res_list.append(sorted_points[i][0])
    return res_list

  def kClosest_oneline(self, points: List[List[int]], k: int) -> List[List[int]]:
    return sorted(points, key=lambda item: math.sqrt(item[0] ** 2 + item[1] ** 2))[:k]


