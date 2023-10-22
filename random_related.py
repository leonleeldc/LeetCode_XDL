'''
528. Random Pick with Weight
You are given a 0-indexed array of positive integers w where w[i] describes the weight of the ith index.

You need to implement the function pickIndex(), which randomly picks an index in the range [0, w.length - 1] (inclusive) and returns it. The probability of picking an index i is w[i] / sum(w).

For example, if w = [1, 3], the probability of picking index 0 is 1 / (1 + 3) = 0.25 (i.e., 25%), and the probability of picking index 1 is 3 / (1 + 3) = 0.75 (i.e., 75%).

'''
from typing import List
from collections import defaultdict
import random
'''
398. Random Pick Index
Given an integer array nums with possible duplicates, randomly output the index of a given target number. You can assume that the given target number must exist in the array.

Implement the Solution class:

Solution(int[] nums) Initializes the object with the array nums.
int pick(int target) Picks a random index i from nums where nums[i] == target. If there are multiple valid i's, then each index should have an equal probability of returning.
Input
["Solution", "pick", "pick", "pick"]
[[[1, 2, 3, 3, 3]], [3], [1], [3]]
Output
[null, 4, 0, 2]

Explanation
Solution solution = new Solution([1, 2, 3, 3, 3]);
solution.pick(3); // It should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
solution.pick(1); // It should return 0. Since in the array only nums[0] is equal to 1.
solution.pick(3); // It should return either index 2, 3, or 4 randomly. Each index should have equal probability of returning.
'''
class RandomPickIndex:
  def __init__(self, nums: List[int]):
    self.nums = nums
    self.cached = defaultdict(list)

  def pick(self, target: int) -> int:
    if target not in self.cached:
      for i in range(len(self.nums)):
        if self.nums[i] == target:
          self.cached[target].append(i)
    return random.choice(self.cached[target])
  # Your Solution object will be instantiated and called as such:
  # obj = Solution(nums)
  # param_1 = obj.pick(target)
  '''
  basic idea is from the Reservoir sampling
  1
  1*1/2, 1/2
  1/2*2/3 = 1/3, 1/2*2/3 = 1/3, 1/3
  for Reservoir sampling, suppose we have 10 balls already in a bag, then the 11th ball comes, 
  It has 10/11 prob to enter into the bag, but we need to remove one from the bag. 
  The removed one has 10/11 * 1/10 = 1/11 to be selected. After that, the 11th one has 10/11 prob into the bag.
  similarly, 12th ball comes, the removed one has 10/12 * 1/10 = 1/12 to be sected, the 12th ball has 11/12 prob into the bag, 
  .., N-1/N.
  product them together, 
  we have 1 * 10/11 * 11/12 * N-1/N = 10/N
  So, we can equal prob to be selected to enter into the bag.
  '''



class RandomPickwithWeight:
  def __init__(self, w: List[int]):
    self.prefix_sums = []
    prefix_sum = 0
    for weight in w:
      prefix_sum += weight
      self.prefix_sums.append(prefix_sum)
    self.total_sum = prefix_sum

  def pickIndex(self) -> int:
    target = self.total_sum * random.random()
    low, high = 0, len(self.prefix_sums) - 1
    while low <= high:
      mid = low + (high - low) // 2
      if target > self.prefix_sums[mid]:
        low = mid + 1
      else:
        high = mid - 1
    return low
'''
138. Copy List with Random Pointer
A linked list of length n is given such that each node contains an additional random pointer, which could point to any node in the list, or null.
Construct a deep copy of the list. The deep copy should consist of exactly n brand new nodes, where each new node has its value set to the value of its corresponding original node. Both the next and random pointer of the new nodes should point to new nodes in the copied list such that the pointers in the original list and copied list represent the same list state. None of the pointers in the new list should point to nodes in the original list.

For example, if there are two nodes X and Y in the original list, where X.random --> Y, then for the corresponding two nodes x and y in the copied list, x.random --> y.

Return the head of the copied linked list.

The linked list is represented in the input/output as a list of n nodes. Each node is represented as a pair of [val, random_index] where:

val: an integer representing Node.val
random_index: the index of the node (range from 0 to n-1) that the random pointer points to, or null if it does not point to any node.
Your code will only be given the head of the original linked list.
'''
# Definition for a Node.
class Node:
    def __init__(self, x: int, next: 'Node' = None, random: 'Node' = None):
        self.val = int(x)
        self.next = next
        self.random = random
class CopyListRandomPointer:
  def copyRandomList(self, head: 'Optional[Node]') -> 'Optional[Node]':
    dummy = new = Node('-1')
    old_new = {}
    cur = head
    while cur:
      new.next = Node(cur.val)
      old_new[cur] = new.next
      cur = cur.next
      new = new.next
    cur = head
    new = dummy.next
    while cur:
      if cur.random:
        new.random = old_new[cur.random]
      cur = cur.next
      new = new.next
    return dummy.next