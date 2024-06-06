'''
109. Convert Sorted List to Binary Search Tree
'''
# Definition for singly-linked list.
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
# Definition for a binary tree node.
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

'''
25. Reverse Nodes in k-Group
Hard
Given the head of a linked list, reverse the nodes of the list k at a time, and return the modified list.
k is a positive integer and is less than or equal to the length of the linked list. If the number of nodes is not a multiple of k then left-out nodes, in the end, should remain as it is.
You may not alter the values in the list's nodes, only nodes themselves may be changed.
'''
class Solution:
  def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
    def reverse(node, k):
      prev, i = None, 0
      while node and k > 0:
        next_node = node.next
        node.next = prev
        prev = node
        node = next_node
        k -= 1
      return prev, node

    dummy = ListNode(0)
    temp1 = head
    dum = dummy
    while temp1:
      temp2 = temp1
      i = 0
      while i < k and temp2:
        temp2 = temp2.next
        i += 1
      if i == k:
        dummy.next, temp1 = reverse(temp1, k)
        while dummy.next:
          dummy = dummy.next
      else:
        dummy.next = temp1
        break
    return dum.next

class SortedList2BST:
  def sortedListToBST_stack(self, head: ListNode) -> TreeNode:
    if not head: return None
    # Find the size of the linked list
    size = 0
    ptr = head
    while ptr:
      size += 1
      ptr = ptr.next
    # Using a stack to emulate the recursive function call stack.
    # (left, right, root) -> left and right as the boundary of linked list,
    # root as the current root of subtree.
    stack = [(0, size - 1, TreeNode(0))]
    dummy_root = stack[0][2]  # Keep the reference to the root.
    # Iterative loop through the stack.
    while stack:
      left, right, root = stack.pop()
      if left <= right:
        mid = (left + right) // 2
        # Move linked list ptr to mid
        ptr = head
        for _ in range(mid):
          ptr = ptr.next
        root.val = ptr.val  # Assign value to the current node.
        # Push the left and right half into stack.
        if left <= mid - 1:
          root.left = TreeNode(0)
          stack.append((left, mid - 1, root.left))
        if mid + 1 <= right:
          root.right = TreeNode(0)
          stack.append((mid + 1, right, root.right))
    return dummy_root

  def sortedListToBST_rec(self, head: ListNode) -> TreeNode:
    def build(subhead):
      slow = fast = subhead
      if not subhead: return None
      mid = 0
      while fast.next and fast.next.next:
        fast = fast.next.next
        slow = slow.next
        mid += 1
      leftnode = ListNode(0)
      dummyNode = leftnode
      for i in range(mid):
        dummyNode.next = ListNode(subhead.val)
        dummyNode = dummyNode.next
        subhead = subhead.next
      root = TreeNode(slow.val)
      root.left = build(leftnode.next)
      root.right = build(slow.next)
      return root
    return build(head)
'''
426. Convert Binary Search Tree to Sorted Doubly Linked List
'''
class BinaryTree2DLL:
  def treeToDoublyList_iter(self, root: 'Optional[Node]') -> 'Optional[Node]':
    if not root: return None
    def dfs(node):
      nonlocal head, tail
      stack = []
      while stack or node:
        if node:
          stack.append(node)
          node = node.left
        else:
          node = stack.pop()
          if tail:
            tail.right = node
            node.left = tail
          else:
            head = node
          tail = node
          node = node.right

    head, tail = None, None
    dfs(root)
    head.left = tail
    tail.right = head
    return head
  def treeToDoublyList_rec(self, root: 'Node') -> 'Node':
      if not root: return None
      def dfs(node):
          if not node: return
          nonlocal head, tail
          dfs(node.left)
          if tail:
              tail.right = node
              node.left = tail
          else:
              head = node
          tail = node
          dfs(node.right)
      head, tail = None, None
      dfs(root)
      head.left = tail
      tail.right = head
      return head
'''
114. Flatten Binary Tree to Linked List
Given the root of a binary tree, flatten the tree into a "linked list":

The "linked list" should use the same TreeNode class where the right child pointer points to the next node in the list and the left child pointer is always null.
The "linked list" should be in the same order as a pre-order traversal of the binary tree.
Input: root = [1,2,5,3,4,null,6]
Output: [1,null,2,null,3,null,4,null,5,null,6]
'''
class FlattenBinaryTree2LL:
  def flatten_rec(self, root: Optional[TreeNode]) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    def rec(node):
      if not node: return None
      if not node.left and not node.right: return node
      left = rec(node.left)
      right = rec(node.right)
      if left:
        left.right = node.right
        node.right = node.left
        node.left = None
      return right if right else left
    return rec(root)
    ### 1 is root, then left  = rec(node.left) return 4,
    ### there, left.right = node.right ----> 4.right = 1.right = 5
    ### node.right = node.left -----> return 2
  '''
  the following one is the best one which also aligns to recursive approach. worth understanding
  '''
  def flatten_iter(self, root: Optional[TreeNode]) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    now = root
    while now:
      if now.left:
        pre = now.left
        while pre.right:
          pre = pre.right
        pre.right = now.right
        now.right = now.left
        now.left = None
      now = now.right

  def flatten_stack(self, root: Optional[TreeNode]) -> None:
    """
    Do not return anything, modify root in-place instead.
    """
    if not root: return None
    stack = [root]
    while stack:
      node = stack.pop()
      if node.right:
        stack.append(node.right)
      if node.left:
        stack.append(node.left)
      if stack:
        node.right = stack[-1]
      node.left = None
    '''
    [1]
    1, [5, 2]
    1
    2, [5, 4, 3]
    1, null, 2
    3, [5, 4]
    1, null, 2, null, 3
    4, [5]
    1, null, 2, null, 3, null, 4
    5, [6]
    1, null, 2, null, 3, null, 4, null, 5
    6, []
    1, null, 2, null, 3, null, 4, null, 5, null, 6
    '''

