# Definition for singly-linked list.
from typing import Optional
class Node:
    def __init__(self, val=None, next=None):
        self.val = val
        self.next = next
'''
708 Insert into a Sorted Circular Linked List
Given a Circular Linked List node, which is sorted in non-descending order, write a function to insert a value insertVal into the list such that it remains a sorted circular list. The given node can be a reference to any single node in the list and may not necessarily be the smallest value in the circular list.

If there are multiple suitable places for insertion, you may choose any place to insert the new value. After the insertion, the circular list should remain sorted.

If the list is empty (i.e., the given node is null), you should create a new single circular list and return the reference to that single node. Otherwise, you should return the originally given node.
Input: head = [3,4,1], insertVal = 2
Output: [3,4,1,2]
Explanation: In the figure above, there is a sorted circular list of three elements. You are given a reference to the node with value 3, and we need to insert 2 into the list. The new node should be inserted between node 1 and node 3. After the insertion, the list should look like this, and we should still return node 3.
'''
class InsertIntoSortedCLL:
    def insert(self, head: 'Optional[Node]', insertVal: int) -> 'Node':
        if head is None:
            head = Node(insertVal)
            head.next = head
            return head
        maxnode = walker = head
        while not walker.val <= insertVal <= walker.next.val:
            # print(f'walker val = {walker.val} maxnode val = {maxnode.val}')
            if walker.val > walker.next.val:
                maxnode = walker
            walker = walker.next
            if walker is head: ##for case [1], it will go into this if
                walker = maxnode
                print('does walker become head')
                break
        # print(f'walker val = {walker.val} maxnode val = {maxnode.val}')
        walker.next = Node(insertVal, next=walker.next)
        return head
    '''
    [6,7,8,9,1,2,3,5], 4
    walker val = 6 maxnode val = 6
    walker val = 7 maxnode val = 6
    walker val = 8 maxnode val = 6
    walker val = 9 maxnode val = 6
    walker val = 1 maxnode val = 9
    walker val = 2 maxnode val = 9

    [6,7,8,9,1,2,3,4] 5
    walker val = 6 maxnode val = 6
    walker val = 7 maxnode val = 6
    walker val = 8 maxnode val = 6
    walker val = 9 maxnode val = 6
    walker val = 1 maxnode val = 9
    walker val = 2 maxnode val = 9
    walker val = 3 maxnode val = 9
    walker val = 4 maxnode val = 9
    the following case will go into if walker is head. without it, it will go to infinite loop
    [3, 1], 0 
    [3,0,1]

    '''

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
'''
09. Convert Sorted List to Binary Search Tree
'''
class ConvertSorteList2BST:
    def sortedListToBST(self, head: Optional[ListNode]) -> Optional[TreeNode]:
        def build(subhead):
            fast = slow = subhead
            if not subhead: return None
            mid = 0
            while fast and fast.next:
                slow = slow.next
                fast = fast.next.next
                mid += 1
            leftnode = ListNode(0)
            tmpnode = leftnode
            for i in range(mid):
                tmpnode.next = ListNode(subhead.val)
                tmpnode = tmpnode.next
                subhead = subhead.next
            leftnode = leftnode.next
            root = TreeNode(slow.val)
            root.left = build(leftnode)
            root.right = build(slow.next)
            return root
        return build(head)

class LinkedList:
    '''
    206. Reverse Linked List https://leetcode.com/problems/reverse-linked-list/description/
    '''
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        while head:
            cur = head.next
            head.next = prev
            prev = head
            head = cur
        return prev
    '''
    86. Partition List https://leetcode.com/problems/partition-list/description/
walking example:
[1,4,3,2,5,2]
0. 
l1.next = head = [1,4,3,2,5,2]
l1 = l1.next = [1,4,3,2,5,2]
dummy1 = [0, 1,4,3,2,5,2]
dummy2 = [1]
head = [4,3,2,5,2]
1.
l2.next = head = [4,3,2,5,2]
l2 = l2.next = [3,2,5,2]
dummy1 = [0,1,4,3,2,5,2]
dummy2 = [1,4,3,2,5,2]
head = [3,2,5,2]
2. 
l2.next = head = [3,2,5,2]
l2 = l2.next = [3,2,5,2]
dummy1 = [0,1,4,3,2,5,2]
dummy2 = [1,4,3,2,5,2]
head = [2,5,2]
3. 
l1.next = head = [2,5,2]
l1 = l1.next = [2, 5,2]
dummy1 = [0,1,2,5,2]
dummy2 = [1,4,3,2,5,2]
head = [5,2]
4. 
l2.next = head = [5,2]
l2 = l2.next = [5, 2]
dummy1 = [0,1,2,5,2]
dummy2 = [1,4,3,5,2]
head = [2]
5. 
l1.next = head = [2]
l1 = l1.next = [2]
dummy1 = [0,1,2,2]
dummy2 = [1,4,3,5,2]
head = None
while loop done,
outside: 
l2.next = None --> 
l2 = [5]
dummy2 = [4, 3, 5]
l1.next = dummy2.next --> dummy1 = [1,2,2,4,3,5]
    '''
    def partition(self, head: ListNode, x: int) -> ListNode:
        dummy1 = l1 = ListNode(0)
        dummy2 = l2 = ListNode(1)
        ind = 0
        while head!=None:
            if head.val < x:
                l1.next = head
                l1 = l1.next
            else:
                l2.next = head
                l2 = l2.next
            head = head.next
            print(f'ind = {ind}')
            ind += 1
        l2.next = None
        l1.next = dummy2.next
        return dummy1.next