from DFS import RemoveInvalidParenthesis
from backtrack import WordPattern
from linkedList import LinkedList, ListNode
from dynamic_programming import WordBreakII, StickerToSpellWord
from search_dp_mutual import FindShortestSuperstring, PartitionKEqualSumSubsets
from topological_sorting import TopologicalSorting
from math_related import MathRelated
from operators import ValidParenthesisString
s ="(((((()*)(*)*))())())(()())())))((**)))))(()())()"
vps = ValidParenthesisString()
print(vps.checkValidStringIter(s))
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

