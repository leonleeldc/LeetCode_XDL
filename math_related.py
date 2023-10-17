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
  '''
  973. K Closest Points to Origin
  Given an array of points where points[i] = [xi, yi] represents a point on the X-Y plane and an integer k, return the k closest points to the origin (0, 0).

The distance between two points on the X-Y plane is the Euclidean distance (i.e., √(x1 - x2)2 + (y1 - y2)2).

You may return the answer in any order. The answer is guaranteed to be unique (except for the order that it is in).


  '''
from typing import List
import math
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


