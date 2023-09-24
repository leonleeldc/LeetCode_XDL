# Definition for singly-linked list.
from typing import Optional
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next
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