from typing import List
class Intervals:
  '''
  56. Merge Intervals https://leetcode.com/problems/merge-intervals/description/
  '''
  def merge(self, intervals: List[List[int]]) -> List[List[int]]:
    ans = []
    for interval in sorted(intervals, key=lambda x: x[0]):
      if not ans or interval[0] > ans[-1][1]:
        ans.append(interval)
      else:
        ans[-1][1] = max(ans[-1][1], interval[1])
    return ans

  '''
  scanline method
  '''
  def merge_(self, intervals: List[List[int]]) -> List[List[int]]:
    boundaries = []
    for interval in intervals:
      boundaries.append((interval[0], -1))
      boundaries.append((interval[1], 1))
    boundaries.sort()
    ans = []
    is_matched, left, right = 0
    for boundary in boundaries:
      if is_matched == 0:
        left = boundary[0]
      is_matched += boundary[1]
      if is_matched == 0:
        right = boundary[0]
        ans.append(([left, right]))
    return ans
  '''
  57. Insert Interval
  '''
  def insert_greedy(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    idx, ans = 0, []
    while idx < len(intervals) and intervals[idx][0]<newInterval[0]:
      ans.append(intervals[idx])
      idx += 1
    if not ans or ans[-1][1]<newInterval[0]:
      ans.append(newInterval)
    else:
      ans[-1][1] = max(newInterval[1], ans[-1][1])
    while idx<len(intervals):
      if ans[-1][1] < intervals[idx][0]:
        ans.append(intervals[idx])
      else:
        ans[-1][1] = max(ans[-1][1], intervals[idx][1])
      idx += 1
    return ans

  def insert_binary_search(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
    # Function to find the position to insert the interval using binary search
    def find_position(intervals, new_start):
      low, high = 0, len(intervals)
      while low < high:
        mid = (low + high) // 2
        if intervals[mid][0] < new_start:
          low = mid + 1
        else:
          high = mid
      return low

    # Find the position to insert newInterval
    pos = find_position(intervals, newInterval[0])
    intervals.insert(pos, newInterval)

    # Merge overlapping intervals
    merged = []
    for interval in intervals:
      if not merged or merged[-1][1] < interval[0]:
        merged.append(interval)
      else:
        merged[-1][1] = max(merged[-1][1], interval[1])
    return merged

  '''
  986. Interval List Intersections
  You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [starti, endi] and secondList[j] = [startj, endj]. Each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

A closed interval [a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.
Example 1:
The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of [1, 3] and [2, 4] is [2, 3].
Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]
Example 2:

Input: firstList = [[1,3],[5,9]], secondList = []
Output: []
  '''
  def intervalIntersection(self, firstList: List[List[int]], secondList: List[List[int]]) -> List[List[int]]:
    result = []
    i = j = 0
    while i < len(firstList) and j < len(secondList):
      lo = max(firstList[i][0], secondList[j][0])
      hi = min(firstList[i][1], secondList[j][1])
      if lo <= hi:
        result.append([lo, hi])
      if firstList[i][1] < secondList[j][1]:
        i += 1
      else:
        j += 1
    return result
