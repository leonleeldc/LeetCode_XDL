'''
4. Median of Two Sorted Arrays
two pointer approach
'''
class Solution:
  def findMedianSortedArrays_h2l(self, A: [int], B: [int]) -> float:
    C = [0] * (len(A) + len(B))
    j, k = len(A) - 1, len(B) - 1

    for i in range(len(C) - 1, -1, -1):
      if j == -1 or (k >= 0 and A[j] <= B[k]):
        C[i] = B[k]
        k -= 1
      else:
        C[i] = A[j]
        j -= 1

    if len(C) % 2 == 0:
      median = (C[len(C) // 2 - 1] + C[len(C) // 2]) / 2.0
    else:
      median = C[len(C) // 2]

    return median
  def findMedianSortedArrays_l2h(self, A: [int], B: [int]) -> float:
    C = [0] * (len(A) + len(B))
    j, k = 0, 0

    for i in range(len(C)):
      if j == len(A) or (k < len(B) and A[j] >= B[k]):
        C[i] = B[k]
        k += 1
      else:
        C[i] = A[j]
        j += 1

    if len(C) % 2 == 0:
      median = (C[len(C) // 2 - 1] + C[len(C) // 2]) / 2.0
    else:
      median = C[len(C) // 2]

    return median
