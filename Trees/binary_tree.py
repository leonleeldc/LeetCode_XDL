from typing import Optional, List
import math
from functools import cache
class TreeNode:
  def __init__(self, val=0, left=None, right=None):
    self.val = val
    self.left = left
    self.right = right

def create_binary_tree(nodes, index=0):
  if index < len(nodes):
    if nodes[index] is None:
      return None
    root = TreeNode(nodes[index])
    root.left = create_binary_tree(nodes, 2 * index + 1)
    root.right = create_binary_tree(nodes, 2 * index + 2)
    return root
  return None
'''
2458. Height of Binary Tree After Subtree Removal Queries
You are given the root of a binary tree with n nodes. Each node is assigned a unique value from 1 to n. You are also given an array queries of size m.
You have to perform m independent queries on the tree where in the ith query you do the following:
Remove the subtree rooted at the node with the value queries[i] from the tree. It is guaranteed that queries[i] will not be equal to the value of the root.
Return an array answer of size m where answer[i] is the height of the tree after performing the ith query.
Note:
The queries are independent, so the tree returns to its initial state after each query.
The height of a tree is the number of edges in the longest simple path from the root to some node in the tree.
'''
class HightAfterSubRem:
  def treeQueries(self, root: Optional[TreeNode], queries: List[int]) -> List[int]:
    height = {}
    def dfs(node: Optional[TreeNode]) -> int:
      if not node: return -1
      h = max(dfs(node.left), dfs(node.right)) + 1
      height[node.val] = h
      return h
    dfs(root)
    total_height = height[root.val]
    node = root
    current = 0
    new_height = {}
    '''
    above while loop understanding:
    The goal is to compute the new height of the tree if the subtree 
    rooted at each node in queries is removed. This is achieved using 
    a while loop that updates the new_height dictionary 
    with the new heights after potential removals.
    This is a really smart implementations.
    '''
    while node:
      new_height[node.val] = current
      if node.left and height[node.left.val] + 1 == height[node.val]:
        rightH = height[node.right.val] + 1 if node.right else 0
        if rightH + total_height - height[node.val] > current:
          current = rightH + total_height - height[node.val]
        node = node.left
      else:
        leftH = height[node.left.val] + 1 if node.left else 0
        if leftH + total_height - height[node.val] > current:
          current = leftH + total_height - height[node.val]
        node = node.right
    return [new_height.get(node, total_height) for node in queries]

'''
666. Path Sum IV
If the depth of a tree is smaller than 5, then this tree can be represented by an array of three-digit integers. For each integer in this array:
The hundreds digit represents the depth d of this node where 1 <= d <= 4.
The tens digit represents the position p of this node in the level it belongs to where 1 <= p <= 8. The position is the same as that in a full binary tree.
The units digit represents the value v of this node where 0 <= v <= 9.
Given an array of ascending three-digit integers nums representing a binary tree with a depth smaller than 5, return the sum of all paths from the root towards the leaves.
It is guaranteed that the given array represents a valid connected binary tree.
Example 1
Input: nums = [113,215,221]
Output: 12
Explanation: The tree that the list represents is shown.
The path sum is (3 + 5) + (3 + 1) = 12.
Example 2
Input: nums = [113,221]
Output: 4
Explanation: The tree that the list represents is shown. 
The path sum is (3 + 1) = 4.
'''
class PathSumIV:
  def pathSum(self, nums: List[int]) -> int:
    def getNodeInfo(num):
      return num//100, (num%100)//10, num%10
    def findChildIndex(node_index, child_position):
      depth, _, _ = getNodeInfo(nums[node_index])
      next_depth = depth + 1
      for i in range(node_index + 1, len(nums)):
        d, p, _ = getNodeInfo(nums[i])
        if d == next_depth and p == child_position:
          return i
      return -1
    def dfs(node_index, current_sum):
      if node_index == -1: return 0
      _, pos, value = getNodeInfo(nums[node_index])
      # Calculate positions of left and right children in a full binary tree
      left_child_pos, right_child_pos = 2 * pos - 1, 2*pos
      left_child_index = findChildIndex(node_index, left_child_pos)
      right_child_index = findChildIndex(node_index, right_child_pos)
      if left_child_index == -1 and right_child_index == -1:
        return current_sum + value
      return dfs(left_child_index, current_sum + value) + dfs(right_child_index, current_sum + value)
    return dfs(0, 0)

'''
1104. Path In Zigzag Labelled Binary Tree
In an infinite binary tree where every node has two children, the nodes are labelled in row order.
In the odd numbered rows (ie., the first, third, fifth,...), the labelling is left to right, while in the even numbered rows (second, fourth, sixth,...), the labelling is right to left.
Given the label of a node in this tree, return the labels in the path from the root of the tree to the node with that label.
Example 1:
Input: label = 14
Output: [1,3,4,14]
Example 2:
Input: label = 26
Output: [1,2,6,10,26]
'''
class PathZigzagLabelledBinaryTree:
  def pathInZigZagTree(self, label: int) -> List[int]:
    depth = 0
    node = label
    # Calculate the depth of the node
    while node:
      node >>= 1
      depth += 1

    path = []
    while label:
      path.append(label)
      max_num = 2 ** depth - 1  # Max number at the current depth
      min_num = 2 ** (depth - 1)  # Min number at the current depth
      label = (max_num + min_num - label) // 2  # Flip the label and move to parent
      depth -= 1
    return path[::-1]  # Reverse the path to get from root to the node
  '''
  label = 14
  path = [14]
  depth = 4
  label = (2**4-1+2**3-14)//2 = 4
  path = [14, 4]
  depth = 3
  label = (2**3-1+2**2-4)//2 =3
  path = [14, 4, 3]
  depth = 2
  label = (2**2-1+2**1-3)//2=1
  path = [14, 4, 3, 1]
  '''

'''
979. Distribute Coins in Binary Tree
You are given the root of a binary tree with n nodes where each node in the tree has node.val coins. There are n coins in total throughout the whole tree.
In one move, we may choose two adjacent nodes and move one coin from one node to another. A move may be from parent to child, or from child to parent.
Return the minimum number of moves required to make every node have exactly one coin.
Input: root = [3,0,0]
Output: 2
Explanation: From the root of the tree, we move one coin to its left child, and one coin to its right child.
Input: root = [0,3,0]
Output: 3
Explanation: From the left child of the root, we move two coins to the root [taking two moves]. Then, we move one coin from the root of the tree to the right child.
'''



'''
513. Find Bottom Left Tree Value
Given the root of a binary tree, return the leftmost value in the last row of the tree.
Input: root = [1,2,3,4,null,5,6,null,null,7]
Output: 7
Input: root = [2,1,3]
Output: 1
'''
class FindBottomLeftTreeValue:
  def findBottomLeftValue_iter(self, root: TreeNode) -> int:
    if not root: return 0
    queue = deque([root])
    leftmost = 0
    while queue:
      level_length = len(queue)
      for i in range(level_length):
        node = queue.popleft()
        if i == 0:
          leftmost = node.val  # Record the first node's value of each level
        if node.left:
          queue.append(node.left)
        if node.right:
          queue.append(node.right)
    return leftmost
  def findBottomLeftValue_dfs(self, root: TreeNode) -> int:
      def dfs(node, depth):
          nonlocal max_depth, leftmost
          if not node: return
          # Check if we've reached a new depth
          if depth > max_depth:
              max_depth = depth
              leftmost = node.val
          dfs(node.left, depth + 1)
          dfs(node.right, depth + 1)
      max_depth = 0
      leftmost = root.val
      dfs(root, 1)
      return leftmost

'''
404. Sum of Left Leaves
Given the root of a binary tree, return the sum of all left leaves.
A leaf is a node with no children. A left leaf is a leaf that is the left child of another node.
Example 1:
Input: root = [3,9,20,null,null,15,7]
Output: 24
Explanation: There are two left leaves in the binary tree, with values 9 and 15 respectively.
Example 2:
Input: root = [1]
Output: 0
'''
class SumLeftLeaves:
  def sumOfLeftLeaves(self, root: TreeNode) -> int:
    if not root: return 0
    sum_left = 0
    def dfs(node):
      nonlocal sum_left
      if node.left:
        # Check if left node is a leaf
        if not node.left.left and not node.left.right:
          sum_left += node.left.val
        else:
          dfs(node.left)
      if node.right:
        dfs(node.right)
    dfs(root)
    return sum_left




'''
1530. Number of Good Leaf Nodes Pairs
You are given the root of a binary tree and an integer distance. A pair of two different leaf nodes of a binary tree is said to be good if the length of the shortest path between them is less than or equal to distance.
Return the number of good leaf node pairs in the tree.
Input: root = [1,2,3,null,4], distance = 3
Output: 1
Explanation: The leaf nodes of the tree are 3 and 4 and the length of the shortest path between them is 3. This is the only good pair.


Input: root = [1,2,3,4,5,6,7], distance = 3
Output: 2
Explanation: The good pairs are [4,5] and [6,7] with shortest path = 2. The pair [4,6] is not good because the length of ther shortest path between them is 4.

Input: root = [7,1,4,6,null,5,3,null,null,null,null,null,2], distance = 3
Output: 1
Explanation: The only good pair is [2,5].
'''
class NumberGoodLeafNodesPair:
  def countPairs_optimized(self, root: TreeNode, distance: int) -> int:
    ans = 0
    @cache
    def dfs(node):
      nonlocal ans
      if not node: return []
      if not node.left and not node.right: return [1]
      pls = dfs(node.left)
      prs = dfs(node.right)
      for pl in pls:
        for pr in prs:
          if pl + pr <= distance:
            ans += 1
      return [item + 1 for item in pls + prs if item + 1 < distance]
    dfs(root)
    return ans

  def countPairs_CartesianProduct(self, root, distance):
    def dfs(node):
      f = [0] * (distance + 1)
      if not node: return f, 0
      if not node.left and not node.right:
        f[0] = 1
        return f, 0
      fl, pl = dfs(node.left)
      fr, pr = dfs(node.right)
      pairs = 0
      for dl, cl in enumerate(fl):
        for dr, cr in enumerate(fr):
          if dl + dr + 2 <= distance: pairs += cl * cr  ###Cartesian product
      for i in range(distance):
        f[i + 1] = fl[i] + fr[i]
      return f, pl + pr + pairs

    return dfs(root)[1]


'''
236. Lowest Common Ancestor of a Binary Tree
'''
class LowestCommonAncestorBinaryTree:
  def lowestCommonAncestor_iter1(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
    ###inorder method
    stack = []
    current, curAncester = root, root
    counter, foundFirst = 0, False
    while len(stack) > 0 or current:
      if current:
        stack.append(current)
        current = current.left
      else:
        current = stack.pop()
        if foundFirst and len(stack) < counter:
          curAncester = current
          counter = len(stack)
        if current.val == p.val or current.val == q.val:
          if not foundFirst:
            foundFirst = True
            curAncester = current
            counter = len(stack)
          else:
            return curAncester
        current = current.right
    def lowestCommonAncestor_rec(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        def dfs(node):
            if not node: return None
            if node == p or node == q: return node
            lanc, ranc = dfs(node.left), dfs(node.right)
            if lanc and ranc: return node
            return lanc if lanc else ranc
        return dfs(root)
'''
108. Convert Sorted Array to Binary Search Tree
'''
class TreeConversion:
  def sortedArrayToBST(self, nums):
    if not nums: return None
    mid = len(nums) // 2
    # The root's value is the middle element of the array
    root = TreeNode(nums[mid])
    # Left subtree is built from elements before mid
    root.left = self.sortedArrayToBST(nums[:mid])
    # Right subtree is built from elements after mid
    root.right = self.sortedArrayToBST(nums[mid + 1:])
    return root
  '''
  1382. Balance a Binary Search Tree
  Given the root of a binary search tree, return a balanced binary search tree with the same node values. If there is more than one answer, return any of them.
  A binary search tree is balanced if the depth of the two subtrees of every node never differs by more than 1.
  '''
  def balanceBST(self, root: TreeNode) -> TreeNode:
    def inorder(root, sorted_list):
      if not root: return
      inorder(root.left, sorted_list)
      sorted_list.append(root)
      inorder(root.right, sorted_list)
      return sorted_list

    sorted_list = inorder(root, [])

    def rBuildTree(inorder, il, ir):
      if il > ir: return None
      mid = il + (ir - il) // 2
      cur_root = inorder[mid]
      cur_root.left = rBuildTree(inorder, il, mid - 1)
      cur_root.right = rBuildTree(inorder, mid + 1, ir)
      return cur_root

    cur_root = rBuildTree(sorted_list, 0, len(sorted_list) - 1)
    return cur_root
  '''
  the following is an optimized approach. We may not need to master it.
  '''
  def makeVine(self, grand, count=0):
    node = grand.right
    while node:
      if node.left:
        old_node = node
        node = node.left
        old_node.left = node.right
        node.right = old_node
        grand.right = node
      else:
        count += 1
        grand = node
        node = node.right
    return count

  def compress(self, grand, m):
    node = grand.right
    while m > 1:
      m -= 1
      old_node = node
      node = node.right
      grand.right = node
      old_node.right = node.left
      node.left = old_node
      grand = node
      node = node.right

  def balanceBST_optimized(self, root: TreeNode) -> TreeNode:
    grand = TreeNode()
    grand.right = root
    count = self.makeVine(grand)
    height = int(math.log2(count + 1))
    remaining_nodes = pow(2, height) - 1
    self.compress(grand, count - remaining_nodes)
    while remaining_nodes > 0:
      remaining_nodes /= 2
      self.compress(grand, remaining_nodes)
    return grand.right

class DistributeCoinsInBT:
  def distributeCoins_iter(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    # Stack for modified post-order traversal
    stack = [(root, False)]
    ans = 0
    balance = {}
    while stack:
      node, visited = stack.pop()
      # If the node has been visited once already, compute its balance
      if visited:
        l = balance.get(node.left, 0)
        r = balance.get(node.right, 0)
        subtree_balance = l + r + node.val - 1
        ans += abs(subtree_balance)
        balance[node] = subtree_balance
      # Otherwise, perform post-order traversal
      else:
        stack.append((node, True))
        if node.right: stack.append((node.right, False))
        if node.left: stack.append((node.left, False))
    return ans

  def distributeCoins_rec(self, root: Optional[TreeNode]) -> int:
    ans = 0
    def balance(node):
      nonlocal ans
      if not node: return 0
      # Calculate the balance in the left and right subtrees
      l, r = balance(node.left), balance(node.right)
      # Calculate the total balance for the current subtree
      subtree_balance = l + r + node.val - 1
      # Accumulate the absolute value of the balance as the number of moves
      ans += abs(subtree_balance)
      return subtree_balance
    balance(root)
    return ans

'''
993. Cousins in Binary Tree
Given the root of a binary tree with unique values and the values of two different nodes of the tree x and y, return true if the nodes corresponding to the values x and y in the tree are cousins, or false otherwise.
Two nodes of a binary tree are cousins if they have the same depth with different parents.
Note that in a binary tree, the root node is at the depth 0, and children of each depth k node are at the depth k + 1.
Input: root = [1,2,3,4], x = 4, y = 3
Output: false
Input: root = [1,2,3,null,4,null,5], x = 5, y = 4
Output: true
Constraints:
The number of nodes in the tree is in the range [2, 100].
1 <= Node.val <= 100
Each node has a unique value.
x != y
x and y are exist in the tree.
'''
# Definition for a binary tree node.
from collections import deque, defaultdict
import heapq
from typing import Optional, List

class Node(object):
  def __init__(self, value):
    self.value = value
    self.left = None
    self.right = None
    self.count = 1
  def __str__(self):
    return 'value: {0}, count: {1}'.format(self.value, self.count)

class Tree(object):
  def __int__(self):
    pass
  def insert(self, root, value):
    if not root:
      return Node(value)
    elif root.value == value:
      root.count += 1
    elif value < root.value:
      root.left = self.insert(root.left, value)
    else:
      root.right = self.insert(root.right, value)
    return root

  def create(self, seq):
    root = None
    for word in seq:
      root = self.insert(root, word)
    return root

  def search(self, root, word, depth=1):
    if not root:
      return 0, 0
    elif root.value == word:
      return depth, root.count
    elif word < root.value:
      return self.search(root.left, word, depth + 1)
    else:
      return self.search(root.right, word, depth + 1)


# def print_tree(root):
#   if root:
#     print_tree(root.left)
#     print(root)
#     print_tree(root.right)

# tree = Tree()
# src = ['foo', 'bar', 'foobar', 'bar', 'barfoo']
# tree.create(src)
# print_tree(tree)

# for word in src:
#   print('search {0}, result: {1}'.format(word, tree.search(word)))

'''
124. Binary Tree Maximum Path Sum
A path in a binary tree is a sequence of nodes where each pair of adjacent nodes in the sequence has an edge connecting them. A node can only appear in the sequence at most once. Note that the path does not need to pass through the root.
The path sum of a path is the sum of the node's values in the path.
Given the root of a binary tree, return the maximum path sum of any non-empty path.
'''
class BinaryTreeComputation:
  def find_max_depth(self, root):
    if root is None: return 0
    lh = self.find_max_depth(root.left)
    rh = self.find_max_depth(root.right)
    result = rh+1 if rh>lh else lh+1
    return result

  def maxDepth(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    if root.left is None and root.right is None: return 1
    return 1 + max(self.maxDepth(root.left), self.maxDepth(root.right))
  def maxPathSum_rec(self, root: Optional[TreeNode]) -> int:
    max_sum = float('-inf')
    def dfs(node):
      nonlocal max_sum
      if not node: return 0
      max_l, max_r = max(0, dfs(node.left)), max(0, dfs(node.right))
      max_sum = max(max_sum, max_l+max_r+node.val)
      return max(max_l, max_r)+node.val
    dfs(root)
    return max_sum
  '''
  938. Range Sum of BST
  Given the root node of a binary search tree and two integers low and high, return the sum of values of all nodes with a value in the inclusive range [low, high].
  '''
  def rangeSumBST_rec(self, root: Optional[TreeNode], low: int, high: int) -> int:
    res = 0
    def dfs(node):
      nonlocal res
      if not node: return res
      if low<node.val<high:
        res+=node.val
      dfs(node.left)
      dfs(node.right)
    return dfs(root)
  def rangeSumBST_iter(self, root: TreeNode, low: int, high: int) -> int:
    sum_range = 0
    stack = []
    while stack or root:
      if root:
        stack.append(root)
        root = root.left
      else:
        root = stack.pop()
        if low <= root.val <= high:
          sum_range += root.val
        elif root.val > high:
          break
        root = root.right
    return sum_range
  '''
  124. Binary Tree Maximum Path Sum
  '''
  def maxPathSum_iter(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    max_sum = float('-inf')
    stack1 = [root]
    stack2 = []
    # Iterative post-order traversal using two stacks
    while stack1:
      node = stack1.pop()
      stack2.append(node)
      if node.left:
        stack1.append(node.left)
      if node.right:
        stack1.append(node.right)
    # Track the best path through each node
    best_path_to_node = {None: 0}
    # Second stack for actual dfs computation
    while stack2:
      node = stack2.pop()
      # Logic from your original dfs function
      left_gain = max(0, best_path_to_node.get(node.left, 0))
      right_gain = max(0, best_path_to_node.get(node.right, 0))
      max_sum = max(max_sum, left_gain + right_gain + node.val)
      best_path_to_node[node] = max(left_gain, right_gain) + node.val
    return max_sum
  '''
  671. Second Minimum Node In a Binary Tree
  '''
  def findSecondMinimumValue(self, root: Optional[TreeNode]) -> int:
    first, second = root.val, float('inf')
    def dfs(node):
      nonlocal first, second
      if node:
        if first<node.val<second:
          second = node.val
        elif node.val==first:
          dfs(node.left)
          dfs(node.right)
    dfs(root)
    return second if second < float('inf') else -1
  '''
  270. Closest Binary Search Tree Value
  '''

  def closestValue_rec(self, root: Optional[TreeNode], target: float) -> int:
    def inorder(r: TreeNode):
      return inorder(r.left)+r.val+inorder(r.right) if r.right else []
    return min(inorder(root), key=lambda x: abs(target-x))

  def closestValue_iter(self, root: Optional[TreeNode], target: float) -> int:
    if not root: return float('inf')
    stack = []
    smaller, larger = float('-inf'), float('inf')
    while stack or root:
      if root:
        stack.append(root)
        root = root.left
      else:
        root = stack.pop()
        if root.val >= target:
          larger = min(larger, root.val)
        elif root.val < target:
          smaller = max(smaller, root.val)
        root = root.right
    if larger - target < target - smaller:
      return larger
    else:
      return smaller

  '''
  958. Check Completeness of a Binary Tree
  Given the root of a binary tree, determine if it is a complete binary tree.
  In a complete binary tree, every level, except possibly the last, is completely filled, and all nodes in the last level are as far left as possible. It can have between 1 and 2h nodes inclusive at the last level h.
  '''
  def isCompleteTree_rec(self, root: Optional[TreeNode]) -> bool:
      indices = []  # To store index for each node
      def dfs(node, index):
          if not node: return
          indices.append(index)
          dfs(node.left, 2 * index)
          dfs(node.right, 2 * index + 1)
      dfs(root, 1)
      # Find the maximum index
      max_index = max(indices)
      # Check if we have all indices from 1 to max_index
      for i in range(1, max_index + 1):
          if i not in indices:
              return False
      return True


# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
  def isCompleteTree_bfs(self, root: Optional[TreeNode]) -> bool:
    q = deque([(root, 1)])
    e_id = 1
    while q:
      size = len(q)
      for _ in range(size):
        node, id = q.popleft()
        if id == e_id:
          e_id += 1
        else:
          return False
        if node.left:
          q.append((node.left, id * 2))
        if node.right:
          q.append((node.right, id * 2 + 1))
    return True
  def isCompleteTree_dfs(self, root: Optional[TreeNode]) -> bool:
      self.total_nodes = 0  # Count of total nodes
      self.max_index = 0  # Maximum index observed
      def dfs(node, index):
          if not node: return True
          self.total_nodes += 1
          self.max_index = max(self.max_index, index)
          left_complete = dfs(node.left, 2 * index)
          right_complete = dfs(node.right, 2 * index + 1)
          return left_complete and right_complete
      dfs(root, 1)
      # After DFS, check if max_index is equal to total_nodes
      return self.max_index == self.total_nodes


# Output
# value: bar, count: 2
# value: barfoo, count: 1
# value: foo, count: 1
# value: foobar, count: 1
# search foo, result: (1, 1)
# search bar, result: (2, 2)
# search foobar, result: (2, 1)
# search bar, result: (2, 2)
# search barfoo, result: (3, 1)

class CousinsBinaryTree:
  def __init__(self):
    # To save the depth of the first node.
    self.recorded_depth = None
    self.is_cousin = False
  def isCousins(self, root: TreeNode, x: int, y: int) -> bool:
    # Queue for BFS
    queue = deque([root])
    while queue:
      siblings = False
      cousins = False
      nodes_at_depth = len(queue)
      for _ in range(nodes_at_depth):
        # FIFO
        node = queue.popleft()
        # Encountered the marker.
        # Siblings should be set to false as we are crossing the boundary.
        if node is None:
          siblings = False
        else:
          if node.val == x or node.val == y:
            # Set both the siblings and cousins flag to true
            # for a potential first sibling/cousin found.
            if not cousins:
              siblings, cousins = True, True
            else:
              # If the siblings flag is still true this means we are still
              # within the siblings boundary and hence the nodes are not cousins.
              return not siblings
          queue.append(node.left) if node.left else None
          queue.append(node.right) if node.right else None
          # Adding the null marker for the siblings
          queue.append(None)
      # After the end of a level if `cousins` is set to true
      # This means we found only one node at this level
      if cousins: return False
    return False

  def dfs(self, node, depth, x, y):
    if node is None: return False
    # Don't go beyond the depth restricted by the first node found.
    if self.recorded_depth and depth > self.recorded_depth: return False
    if node.val == x or node.val == y:
      if self.recorded_depth is None:
        # Save depth for the first node.
        self.recorded_depth = depth
      # Return true, if the second node is found at the same depth.
      return self.recorded_depth == depth
    left = self.dfs(node.left, depth + 1, x, y)
    right = self.dfs(node.right, depth + 1, x, y)
    # self.recorded_depth != depth + 1 would ensure node x and y are not
    # immediate child nodes, otherwise they would become siblings.
    if left and right and self.recorded_depth != depth + 1:
      self.is_cousin = True
    return left or right

  def isCousins_dfs(self, root: TreeNode, x: int, y: int) -> bool:
    # Recurse the tree to find x and y
    self.dfs(root, 0, x, y)
    return self.is_cousin

  def isCousins_bfs(self, root: Optional[TreeNode], x: int, y: int) -> bool:
    queue = deque([(root, 0)])
    x_level, x_parent, y_level, y_parent = 0, None, 0, None
    while queue:
      size = len(queue)
      for _ in range(size):
        node, level = queue.popleft()
        if node.left:
          if node.left.val == x:
            x_level = level + 1
            x_parent = node.val
          if node.left.val == y:
            y_level = level + 1
            y_parent = node.val
          queue.append((node.left, level + 1))
        if node.right:
          if node.right.val == x:
            x_level = level + 1
            x_parent = node.val
          if node.right.val == y:
            y_level = level + 1
            y_parent = node.val
          queue.append((node.right, level + 1))
    return x_parent != y_parent and x_level == y_level

'''
199. Binary Tree Right Side View
'''
class BinaryTreeRsv:
  def rightSideView_rec(self, root: Optional[TreeNode]) -> List[int]:
    if not root: return []
    res = []
    def dfs(node, depth):
      if not node: return
      if depth == len(res):
        res.append(node.val)
      if node.right:
        dfs(node.right, depth+1)
      if node.left:
        dfs(node.left, depth+1)
    dfs(root, 0)
    return res
  def rightSideView_iter(self, root: Optional[TreeNode]) -> List[int]:
    if not root: return []
    queue = deque(([root, 0]))
    res = []
    while queue:
      node, depth = queue.popleft()
      if depth == len(res):
        res.append(node.val)
      if node.left:
        queue.append([root.left, depth+1])
      if node.right:
        queue.append([root.right, depth-1])
    return res

  def rightSideView_bfs(self, root: Optional[TreeNode]) -> List[int]:
    if not root: return []
    queue = deque([root])
    output = []
    while queue:
      size = len(queue)
      for i in range(size):
        node = queue.popleft()
        if i == size - 1:
          output.append(node.val)
        if node.left: queue.append(node.left)
        if node.right: queue.append(node.right)
    return output

'''
1457 Pseudo-Palindromic Paths in a Binary Tree
Given a binary tree where node values are digits from 1 to 9. A path in the binary tree is said to be pseudo-palindromic if at least one permutation of the node values in the path is a palindrome.
Return the number of pseudo-palindromic paths going from the root node to leaf nodes.
'''
class PseudoPalinPathBT:
  def pseudoPalindromicPaths_rec1(self, root: Optional[TreeNode]) -> int:
    counts = [0] * 10
    def dfs(node):
      if not node: return 0
      counts[node.val] += 1
      c = 0
      if not node.left and not node.right:
        odds = 0
        for i in range(10):
          if counts[i] & 1: odds += 1
        if odds <= 1: c = 1
      l = dfs(node.left)
      r = dfs(node.right)
      counts[node.val] -= 1  # back tracking
      return c + l + r
    return dfs(root)
  def pseudoPalindromicPaths_rec2(self, root: Optional[TreeNode]) -> int:
    def dfs(node, s):
      if not node: return 0
      # Update the count parity for the current digit.
      s ^= (1 << node.val)
      ans = 0
      # If it is a leaf node.
      if not node.left and not node.right:
        # Check if at most one bit is set in s. If so, it's a pseudo-palindromic path.
        if s & (s - 1) == 0: ans += 1
      else:
        # Recursively check the left and right children.
        ans += dfs(node.left, s)
        ans += dfs(node.right, s)
      # Backtrack implicitly happens here as s is not modified in the parent call.
      return ans
    # Start the DFS from the root with an initial state of 0.
    return dfs(root, 0)
  def pseudoPalindromicPaths_iter(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    stack = [(root, 0)]
    ans = 0
    while stack:
      node, s = stack.pop()
      s ^= (1<<node.val)
      if not node.left and not node.right:
        if s & (s-1) == 0: ans +=1
      else:
        if node.left: stack.append((node.left, s))
        if node.right: stack.append((node.right, s))
    return ans

  def pseudoPalindromicPaths_bfs(self, root: Optional[TreeNode]) -> int:
    if not root: return 0
    ans = 0
    # Initialize a queue for BFS. Each item is a tuple (node, s),
    # where node is the current node and s is an integer representing
    # the parity of the path from the root to the node.
    queue = deque([(root, 0)])
    while queue:
      # Dequeue the node and state.
      node, s = queue.popleft()
      # Update the count parity for the current digit.
      s ^= (1 << node.val)
      # Check if it is a leaf node.
      if not node.left and not node.right:
        # Check if at most one bit is set in s. If so, it's a pseudo-palindromic path.
        if s & (s - 1) == 0: ans += 1
      else:
        # Add the left child to the queue with the updated state.
        if node.left: queue.append((node.left, s))
        # Add the right child to the queue with the updated state.
        if node.right: queue.append((node.right, s))
    return ans
  def verticalTraversal(self, root: Optional[TreeNode]) -> List[List[int]]:
      '''
      987. Vertical Order Traversal of a Binary Tree
      Given the root of a binary tree, calculate the vertical order traversal of the binary tree.
      For each node at position (row, col), its left and right children will be at positions (row + 1, col - 1) and (row + 1, col + 1) respectively. The root of the tree is at (0, 0).
      The vertical order traversal of a binary tree is a list of top-to-bottom orderings for each column index starting from the leftmost column and ending on the rightmost column. There may be multiple nodes in the same row and same column. In such a case, sort these nodes by their values.
      Return the vertical order traversal of the binary tree.
      '''
      queue = deque([(root, 0, 0)])
      col_dict = defaultdict(list)
      heapq.heappush(col_dict[0], (0, root.val))
      while queue:
          size = len(queue)
          for _ in range(size):
              node, col, row = queue.popleft()
              if node.left:
                  heapq.heappush(col_dict[col-1], (row+1, node.left.val))
                  queue.append((node.left, col-1, row+1))
              if node.right:
                  heapq.heappush(col_dict[col+1], (row+1, node.right.val))
                  queue.append((node.right, col+1, row+1))
      res = []
      min_c, max_c = min(col_dict), max(col_dict)
      for i in range(min_c, max_c+1):
          col_res = []
          while col_dict[i]: col_res.append(heapq.heappop(col_dict[i])[1])
          res.append(col_res)
      return res

  def verticalTraversal_dfs(self, root: TreeNode) -> List[List[int]]:
    from heapq import heappush, heappop
    col_dict = defaultdict(list)
    heappush(col_dict[0], (0, root.val))

    def traverse(node, row, col):
      if node.left:
        heappush(col_dict[col - 1], (row + 1, node.left.val))
        traverse(node.left, row + 1, col - 1)
      if node.right:
        heappush(col_dict[col + 1], (row + 1, node.right.val))
        traverse(node.right, row + 1, col + 1)

    traverse(root, 0, 0)

    res = []
    min_c, max_c = min(col_dict), max(col_dict)
    for i in range(min_c, max_c + 1):
      col_res = []
      while col_dict[i]: col_res.append(heappop(col_dict[i])[1])
      res.append(col_res)
    return res

  '''
  662. Maximum Width of Binary Tree
  Given the root of a binary tree, return the maximum width of the given tree.
  The maximum width of a tree is the maximum width among all levels.
  The width of one level is defined as the length between the end-nodes (the leftmost and rightmost non-null nodes), where the null nodes between the end-nodes that would be present in a complete binary tree extending down to that level are also counted into the length calculation.
  It is guaranteed that the answer will in the range of a 32-bit signed integer.
  '''
  def widthOfBinaryTree_rec(self, root: Optional[TreeNode]) -> int:
    max_width = 0
    depth_dict = {}
    def dfs(node, depth, col_ind):
      nonlocal max_width
      if not node: return
      if depth not in depth_dict:
        depth_dict[depth] = col_ind
      max_width = max(max_width, col_ind - depth_dict[depth] + 1)
      dfs(node.left, depth + 1, col_ind * 2)
      dfs(node.right, depth + 1, col_ind * 2 + 1)
    dfs(root, 0, 0)
    return max_width
  def widthOfBinaryTree_iter(self, root: Optional[TreeNode]) -> int:
    queue = deque([(root, 0)])
    max_width = 0
    while queue:
      _, left_ind = queue[0]
      size = len(queue)
      for _ in range(size):
        node, col_ind = queue.popleft()
        if node.left: queue.append((node.left, col_ind * 2))
        if node.right: queue.append((node.right, col_ind * 2 + 1))
      max_width = max(max_width, col_ind - left_ind + 1)
    return max_width



