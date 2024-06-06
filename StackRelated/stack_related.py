'''
716. Max Stack
Design a max stack data structure that supports the stack operations and supports finding the stack's maximum element.

Implement the MaxStack class:

MaxStack() Initializes the stack object.
void push(int x) Pushes element x onto the stack.
int pop() Removes the element on top of the stack and returns it.
int top() Gets the element on the top of the stack without removing it.
int peekMax() Retrieves the maximum element in the stack without removing it.
int popMax() Retrieves the maximum element in the stack and removes it. If there is more than one maximum element, only remove the top-most one.
You must come up with a solution that supports O(1) for each top call and O(logn) for each other call.



Example 1:

Input
["MaxStack", "push", "push", "push", "top", "popMax", "top", "peekMax", "pop", "top"]
[[], [5], [1], [5], [], [], [], [], [], []]
Output
[null, null, null, null, 5, 5, 1, 5, 1, 5]

Explanation
MaxStack stk = new MaxStack();
stk.push(5);   // [5] the top of the stack and the maximum number is 5.
stk.push(1);   // [5, 1] the top of the stack is 1, but the maximum is 5.
stk.push(5);   // [5, 1, 5] the top of the stack is 5, which is also the maximum, because it is the top most one.
stk.top();     // return 5, [5, 1, 5] the stack did not change.
stk.popMax();  // return 5, [5, 1] the stack is changed now, and the top is different from the max.
stk.top();     // return 1, [5, 1] the stack did not change.
stk.peekMax(); // return 5, [5, 1] the stack did not change.
stk.pop();     // return 1, [5] the top of the stack and the max element is now 5.
stk.top();     // return 5, [5] the stack did not change.

'''

import heapq
class MaxStackOptimized:

  def __init__(self):
    self.stack = []
    self.minHeap = []
    self.idx = 0
    self.removed = set()

  def push(self, x: int) -> None:
    self.stack.append((x, self.idx))
    heapq.heappush(self.minHeap, (-x, -self.idx))
    self.idx += 1

  def pop(self) -> int:
    while self.stack and self.stack[-1][1] in self.removed:
      self.stack.pop()
    val, i = self.stack.pop()
    self.removed.add(i)
    return val

  def top(self) -> int:
    while self.stack and self.stack[-1][1] in self.removed:
      self.stack.pop()
    return self.stack[-1][0]

  def peekMax(self) -> int:
    while self.minHeap and -self.minHeap[0][1] in self.removed:
      heapq.heappop(self.minHeap)
    return -self.minHeap[0][0]

  def popMax(self) -> int:
    while self.minHeap and -self.minHeap[0][1] in self.removed:
      heapq.heappop(self.minHeap)
    val, i = heapq.heappop(self.minHeap)
    self.removed.add(-i)
    return -val
class MaxStack:

  def __init__(self):
    self.stack = []
    self.switch_stack = []
    self.max = float('-inf')
    self.max_stack = []

  def push(self, x: int) -> None:
    self.max = max(self.max, x)
    self.stack.append(x)

  def pop(self) -> int:
    ans = self.stack.pop()
    if self.max == ans:
      while self.stack and self.stack[-1]!=self.max:
        self.max = float('-inf')
        self.max = max(self.max, self.stack[-1])
        self.switch_stack.append(self.stack.pop())
      while self.switch_stack:
        self.stack.append(self.switch_stack.pop())
    return ans

  def top(self) -> int:
    return self.stack[-1]

  def peekMax(self) -> int:
    return self.max

  def popMax(self) -> int:
    ans = self.max
    while self.stack and self.stack[-1] != self.max:
      self.max = max(self.max, self.stack[-1])
      self.switch_stack.append(self.stack.pop())
    self.stack.pop()
    self.max = float('-inf')
    while self.stack:
      self.max = max(self.max, self.stack[-1])
      self.switch_stack.append(self.stack.pop())
    while self.switch_stack:
      self.max = max(self.max, self.switch_stack[-1])
      self.stack.append(self.switch_stack.pop())
    return ans

'''
895. Maximum Frequency Stack
Design a stack-like data structure to push elements to the stack and pop the most frequent element from the stack.

Implement the FreqStack class:

FreqStack() constructs an empty frequency stack.
void push(int val) pushes an integer val onto the top of the stack.
int pop() removes and returns the most frequent element in the stack.
If there is a tie for the most frequent element, the element closest to the stack's top is removed and returned.
'''
from collections import defaultdict
class FreqStack:

  def __init__(self):
    self.items = defaultdict(int)  # Sotre count of ea val, aka lvl
    self.lvls = defaultdict(list)
    self.max_lvl = 1

  def push(self, val: int) -> None:
    self.items[val] += 1
    lvl = self.items[val]
    self.lvls[lvl].append(val)
    self.max_lvl = max(self.max_lvl, lvl)

  def pop(self) -> int:
    max_row_len = len(self.lvls[self.max_lvl])
    item = self.lvls[self.max_lvl].pop()
    self.items[item] -= 1
    if max_row_len == 1:  # We know the list will be empty now
      self.max_lvl -= 1
    return item