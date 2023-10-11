'''
373. Find K Pairs with Smallest Sums
'''
from typing import List
import heapq
class HeapRelated:
  def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    ans, heap = [], []  # initialize variables
    visited = set()  # to keep a check whether index i,j are already visited
    i, j = 0, 0
    visited.add((i, j))  # add the i,j index to the visited
    heapq.heappush(heap, (nums1[i] + nums2[j], i, j))  # push the sum(nums1[i],nums2[j]) and i,j into the heap
    while k > 0 and heap:  # keep the loop untill k element are found or all the element are travelled
      _, i, j = heapq.heappop(heap)  # pop the element from the heap and append it to the answer
      ans.append([nums1[i], nums2[j]])
      if (i + 1) < len(nums1) and (i + 1,
                                   j) not in visited:  # check if (i + 1) < len(nums1) and (i+1),j are not in the visited then add the sum of i+1,j and i+1,j to the heap and visited
        heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
        visited.add((i + 1, j))
      if (j + 1) < len(nums2) and (i, j + 1) not in visited:  # similarly as above check for the (j + 1)
        heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
        visited.add((i, j + 1))
      k -= 1  # decrement the k
    return ans