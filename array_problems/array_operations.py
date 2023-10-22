'''
341. Flatten Nested List Iterator
You are given a nested list of integers nestedList. Each element is either an integer or a list whose elements may also be integers or other lists. Implement an iterator to flatten it.

Implement the NestedIterator class:

NestedIterator(List<NestedInteger> nestedList) Initializes the iterator with the nested list nestedList.
int next() Returns the next integer in the nested list.
boolean hasNext() Returns true if there are still some integers in the nested list and false otherwise.
Your code will be tested with the following pseudocode:

initialize iterator with nestedList
res = []
while iterator.hasNext()
    append iterator.next() to the end of res
return res
If res matches the expected flattened list, then your code will be judged as correct.



Example 1:

Input: nestedList = [[1,1],2,[1,1]]
Output: [1,1,2,1,1]
Explanation: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,1,2,1,1].
Example 2:

Input: nestedList = [1,[4,[6]]]
Output: [1,4,6]
Explanation: By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,4,6].
'''
class NestedIterator:
  def __init__(self, nestedList: [NestedInteger]):
    self.stack = []
    # Add the nestedList to the stack in reverse order
    for i in range(len(nestedList) - 1, -1, -1):
      self.stack.append(nestedList[i])

  def next(self) -> int:
    return self.stack.pop().getInteger()

  def hasNext(self) -> bool:
    # Flatten the list by popping elements from the stack until we find an integer
    while self.stack:
      current = self.stack[-1]
      if current.isInteger():
        return True

      # If it's a list, pop it and push its elements in reverse order
      self.stack.pop()
      nested_list = current.getList()
      for i in range(len(nested_list) - 1, -1, -1):
        self.stack.append(nested_list[i])

    return False

  class NestedIteratorRec:
    def __init__(self, nestedList):
      self.stack = []
      self.flatten(nestedList)

    def flatten(self, nestedList):
      for item in reversed(nestedList):
        if item.isInteger():
          self.stack.append(item.getInteger())
        else:
          self.flatten(item.getList())

    def next(self):
      return self.stack.pop()

    def hasNext(self):
      return len(self.stack) > 0


def flat(arr, n):
  ans = []
  def dfs(depth, maxDepth, currArray):
    if depth == maxDepth:
      ans.append(currArray)
      return
    for element in currArray:
      if isinstance(element, list):
        dfs(depth + 1, maxDepth, element)
      else:
        ans.append(element)
  dfs(-1, n, arr)
  return ans
