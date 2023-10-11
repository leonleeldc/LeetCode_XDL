'''
109. Convert Sorted List to Binary Search Tree
'''
# Definition for singly-linked list.
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

    return build(head)



