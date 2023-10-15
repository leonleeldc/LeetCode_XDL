from StringProblems.string_problems import StringProblemsClass
spc = StringProblemsClass()
s = "the sky is blue"
print(spc.reverseWords(s))
from Trees.binary_tree import TreeNode
tree_node = TreeNode(-10)
tree_node.left = TreeNode(9)
tree_node.right = TreeNode(20)
tree_node.right.left = TreeNode(15)
tree_node.right.right = TreeNode(7)
from Trees.binary_tree import BinaryTreeComputation
btc = BinaryTreeComputation()
print(btc.maxPathSum_iter(tree_node))
# for A in range(1, 10):
#     for B in range(1, 10):
#         if 100*A*B + 10*A**2 + 49*B + 6*A == 2900:
#             print(f"A = {A}, B = {B}")
#             break
from array_problems.sub_array_problems import ShortestSubArrMaxsum
from StringProblems.string_comparisons import StringProblemClass
# s = "ADOBECODEBANC"
# chars = "ABC"
# sp = StringProblemClass()
# # chars = ["a","a","b","b","c","c","c"]
# # print(sp.compress(chars))
# chars = ["a","b","b","b","b","b","b","b","b","b","b","b","b"]
# print(sp.compress(chars))
# ss = 'abcabcbb'
# print(sp.lengthOfLongestSubstring(ss))
# print(sp.shortest_substring(s, chars))  # Output: "BANC"

# Examples
##the following is not correct yet
# ssam = ShortestSubArrMaxsum()
# print(ssam.shortest_subarray_with_max_sum([1, 2, -3, 4, 5, -10, 9]))  # Output: 1
# print(ssam.shortest_subarray_with_max_sum([1,-1,1,1, 1,-1,1,1,-2,2])) ## Output: 3
# print(ssam.shortest_subarray_with_max_sum([1]))  # Output: 1
# print(ssam.shortest_subarray_with_max_sum([1, 2, 3, 4]))  # Output: 4
# print(ssam.shortest_subarray_with_max_sum([]))  # Output: 0
# print('test')
# s ="(((((()*)(*)*))())())(()())())))((**)))))(()())()"
# vps = ValidParenthesisString()
# print(vps.checkValidStringIter(s))
# nums = [4,3,2,3,5,2,1]
# k = 4
# pkess = PartitionKEqualSumSubsets()
# print(pkess.canPartitionKSubsets_dp1(nums=nums, k=k))
# print(pkess.canPartitionKSubsets_dp2(nums=nums, k=k))
# fss = FindShortestSuperstring()
# input = ["alex","loves","leetcode"]
# print(fss.shortestSuperstring_dp_no_permutations(input))

# sts = StickerToSpellWord()
# stickers = ["with","example","science"]
# target = "thehat"
# sts.minStickers(stickers, target=target)

# math_prob = MathRelated()
# x = 0
# math_prob.reverse(x)
# x = 123
# math_prob.reverse(x)
# x = -123
# math_prob.reverse(x)
# x = 120
# math_prob.reverse(x)

#words = ["wrt","wrf","er","ett","rftt"]
# words = ["z","x","z"]
# top_sort = TopologicalSorting()
# print(top_sort.alienOrder_bfs(words))
# print(top_sort.alienOrder_dfs(words))
# s = "catsanddog"
# wordDict = ["cat","cats","and","sand","dog"]
# wb2 = WordBreakII()
# output = wb2.wordBreak_iter(wordDict=wordDict, s=s)
# output2 = wb2.wordBreak_rec(wordDict=wordDict, s=s)
# print(output)
# input = [1,4,3,2,5,2]
# x = 3
# #Output: [1,2,2,4,3,5]
# head = ListNode(input[0])
# walker = head
# for val in input[1:]:
#   walker.next = ListNode(val)
#   walker = walker.next
# ll = LinkedList()
# par = ll.partition(head, 3)
# print(par)
# input = [1,2,3,4,5]
# head = ListNode(input[0])
# walker = head
# for val in input[1:]:
#   walker.next = ListNode(val)
#   walker = walker.next
# ll = LinkedList()
# revLL = ll.reverseList(head)
# print(revLL)


# wp1 = WordPattern()
# pattern = "abab"
# str = "redblueredblue"
# print(f"solution = {wp1.backtrack(pattern, str, {})}")

# from DFS import RemoveInvalidParenthesis
# rip = RemoveInvalidParenthesis()
# '''
# Example 1:
#
# Input: s = "()())()"
# Output: ["(())()","()()()"]
# Example 2:
#
# Input: s = "(a)())()"
# Output: ["(a())()","(a)()()"]
# Example 3:
#
# Input: s = ")("
# Output: [""]
# '''
# s = "()())()"
# print(rip.remove_invalid_parentheses(s))

