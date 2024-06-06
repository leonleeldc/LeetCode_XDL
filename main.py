from BackTrack.backtrack import SynonymousSentencesV2
ss = SynonymousSentencesV2()
synonyms = [["happy","joy"],["sad","sorrow"],["joy","cheerful"]]
text = "I am happy today but was sad yesterday"
ss.generateSentences(synonyms, text)

# from BackTrack.backtrack import SynonymousSentences
# ss = SynonymousSentences()
# synonyms = [["happy","joy"],["sad","sorrow"],["joy","cheerful"]]
# text = "I am happy today but was sad yesterday"
# ss.generateSentences(synonyms, text)
# from heap_related import ConcatenatedWords
# ccw = ConcatenatedWords()
# words = ["cat","cats","catsdogcats","dog","dogcatsdog","hippopotamuses","rat","ratcatdogcat"]
# print(ccw.findAllConcatenatedWordsInADict(words))
# from BinarySearch.binary_search_related import Searc2DhMat
# search2DMat = Searc2DhMat()
# matrix = [[1,3,5,7],[10,11,16,20],[23,30,34,60]]
# target = 13
# print(search2DMat.searchMatrix(matrix, target))
# from StackRelated.stack_related import MaxStack
# maxStack = MaxStack()
# maxStack.push(5)
# maxStack.push(1)
# # maxStack.push(5)
# # print(maxStack.top())
# print(maxStack.popMax())
# # print(maxStack.top())
# print(maxStack.peekMax())
# # print(maxStack.pop())
# # print(maxStack.top())
# print('done')
# from StackRelated.stack_related import FreqStack
# freq_stack = FreqStack()
# input = [5, 7, 5, 7, 4, 5]
# for item in input:
#   freq_stack.push(item)
# for i in range(4):
#   freq_stack.pop()

# from SortingProblems.sorting_related import CountSmallerNumbersAfterSelf
# csnas = CountSmallerNumbersAfterSelf()
# nums = [5,2,6,1]
# print(csnas.countSmaller_bt(nums))
# from DynamicProgramming.dynamic_programming import OddEvenJump
# oej = OddEvenJump()
# arr = [10,13,12,14,15]
# oej.oddEvenJumps(arr)
# from SlideWindows.counts import SubWithKDiffInts
# swkd = SubWithKDiffInts()
# nums = [1,2,1,2,3]
# k = 2
# print(swkd.subarraysWithKDistinct(nums, k))
# from GraphProblems.graph import NumCoinsPlacingInTreeNode
# ncpit = NumCoinsPlacingInTreeNode()
# edges = [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[0,8],[0,9],[0,10],[0,11],[0,12],[0,13],[0,14],[0,15],[0,16],[0,17],[0,18],[0,19],[0,20],[0,21],[0,22],[0,23],[0,24],[0,25],[0,26],[0,27],[0,28],[0,29],[0,30],[0,31],[0,32],[0,33],[0,34],[0,35],[0,36],[0,37],[0,38],[0,39],[0,40],[0,41],[0,42],[0,43],[0,44],[0,45],[0,46],[0,47],[0,48],[0,49],[0,50],[0,51],[0,52],[0,53],[0,54],[0,55],[0,56],[0,57],[0,58],[0,59],[0,60],[0,61],[0,62],[0,63],[0,64],[0,65],[0,66],[0,67],[0,68],[0,69],[0,70],[0,71],[0,72],[0,73],[0,74],[0,75],[0,76],[0,77],[0,78],[0,79],[0,80],[0,81],[0,82],[0,83],[0,84],[0,85],[0,86],[0,87],[0,88],[0,89],[0,90],[0,91],[0,92],[0,93],[0,94],[0,95],[0,96],[0,97],[0,98],[0,99]]
# cost = [-5959,602,-6457,7055,-1462,6347,7226,-8422,-6088,2997,-7909,6433,5217,3294,-3792,7463,8538,-3811,5009,151,5659,4458,-1702,-1877,2799,9861,-9668,-1765,2181,-8128,7046,9529,6202,-8026,6464,1345,121,1922,7274,-1227,-9914,3025,1046,-9368,-7368,6205,-6342,8091,-6732,-7620,3276,5136,6871,4823,-1885,-4005,-3974,-2725,-3845,-8508,7201,-9566,-7236,-3386,4021,6793,-8759,5066,5879,-5171,1011,1242,8536,-8405,-9646,-214,2251,-9934,-8820,6206,1006,1318,-9712,7230,5608,-4601,9185,346,3056,8913,-2454,-3445,-4295,4802,-8852,-6121,-4538,-5580,-9246,-6462]
# edges = [[0,1],[0,2],[1,3],[1,4],[1,5],[2,6],[2,7],[2,8]]
# cost = [1,4,2,3,5,7,8,-4,2]
# edges = [[0,2],[0,6],[1,4],[3,5],[7,6],[3,6],[1,8],[3,1],[9,3]]
# cost = [63,13,-6,20,56,-14,61,25,-99,54]
# print(ncpit.placedCoins(edges, cost))
# print(ncpit.placedCoinsBFS(edges, cost)) ### order = [0, 2, 6, 7, 3, 5, 1, 9, 4, 8] rather than [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

'''
[215208, 0, 1, 77616, 1, 1, 184464, 1, 1, 1]
if we do not run first BFS, directly use 
[96075, 184464, 1, 82350, 1, 1, 1, 1, 1, 1]
'''

# from Trees.binary_tree import HightAfterSubRem, create_binary_tree
# # Helper function to create a binary tree from a list
# # Example usage:
# nodes = [5, 8, 9, 2, 1, 3, 7, 4, 6]
# queries = [3, 2, 4, 8]
# root = create_binary_tree(nodes)
# hasr = HightAfterSubRem()
# print(hasr.treeQueries(root, queries))  # Output should be [3, 2, 3, 2]

# from SystemOp.cache import MRUQueue
# # Example usage
# mru_queue = MRUQueue(8)
# print(mru_queue.fetch(3))  # Moves 3rd element to the end and returns it
# print(mru_queue.fetch(5))  # Moves 5th element to the end and returns it

# from operators import RLEIterator
# int_list = [3, 8, 0, 9, 2, 5]
# rleIter = RLEIterator(int_list)
# rleIter

# from operators import RunLengthIterator, ChunkReceiver
# # Example usage
# encoded = [1, 2, 3, 4, 1, 5]
# iterator = RunLengthIterator(encoded)
# unpacked = list(iterator)
# print(unpacked)  # Output: [2, 4, 4, 4, 5]

# Example usage
# receiver = ChunkReceiver()
# receiver.add_chunk(0, 'data0')
# print(receiver.get_expected_chunk())  # Output: 1
# receiver.add_chunk(2, 'data2')
# print(receiver.get_expected_chunk())  # Output: 1
# receiver.add_chunk(1, 'data1')
# print(receiver.get_expected_chunk())  # Output: 3
# from heap_related import SingleThreadedCPU
# stc = SingleThreadedCPU()
# # tasks = [[1,2],[2,4],[3,2],[4,1]]
# tasks=[[2, 10], [0, 5], [1, 9],  [2, 5], [4, 7], [6, 7]]
# stc.getOrder(tasks)
# from topological_sorting import CourseSchedule
# courseScheduler = CourseSchedule()
# numCourses = 2
# prerequisites = [[1, 0]]
# print(courseScheduler.canFinish(numCourses, prerequisites))
# from random_related import CopyListRandomPointer, Node
# #head = [[7,null],[13,0],[11,4],[10,2],
# rand_node = Node(17)
# rand_node.next = Node(13, random=Node(0))
# rand_node.next.next = Node(11, random=Node(4))
# rand_node.next.next.next = Node(10, random=Node(2))
# crp = CopyListRandomPointer()
# crp.copyRandomList(rand_node)
# from DynamicProgramming.dynamic_programming import MaxSum3NonOverlappingSubArrays
# nums = [1,2,1,2,6,7,5,1]
# k = 2
# max_sum_subarray = MaxSum3NonOverlappingSubArrays()
# max_sum_subarray.maxSumOfThreeSubarrays(nums,k=2)
# print(max_sum_subarray)
# from TwoPointers.two_pointers import TwoPointersRelated
# tpr = TwoPointersRelated()
# push_lst = [1, 2, 3]
# pop_lst = [3, 2, 1]
# tpr.detect_stack_op(push_lst, pop_lst)
# push_lst = [1, 2, 3]
# pop_lst = [3, 1, 2]
# tpr.detect_stack_op(push_lst, pop_lst)
# push_lst = [1, 2, 3]
# pop_lst = [2, 1, 3]
# tpr.detect_stack_op(push_lst, pop_lst)
#from array_problems.array_operations import TicTacToe
#["TicTacToe", "move", "move", "move", "move", "move", "move", "move"]
#[[3], [0, 0, 1], [0, 2, 2], [2, 2, 1], [1, 1, 2], [2, 0, 1], [1, 0, 2], [2, 1, 1]]
# ttt = TicTacToe(3)
# ttt.move(0, 0, 1)
# ttt.move(0, 2, 2)
# ttt.move(2, 2, 1)
# ttt.move(1, 1, 2)
# ttt.move(2, 0, 1)
# ttt.move(1, 0, 2)
# ttt.move(2, 1, 1)
# print('done')
# from math_related import combine_coins, math_approch
# combine_coins()
# math_approch()
# from SlideWindows.counts import MoveZeros
# mz = MoveZeros()
# nums = [2, 0, 0, 0, 3, 0, 0, 5]
# k = 1
# mz.moveZeroesWithGap(nums, k)
#print(nums)  # For k = 1, the output will be [2, 0, 3, 0, 5, 0, 0, 0]


# from array_problems.array_operations import KSmallestPairDistance
# kspd = KSmallestPairDistance()
# nums = [1,6,1]
# k = 3
# kspd.smallestDistancePair(nums, k)
# from array_problems.sum_series import CountIncreasingQuadruplets
# nums = [1,3,2,4,5]
# ciq = CountIncreasingQuadruplets()
# ciq.countQuadruplets(nums)
# nums = [1,3,2,4,5, 6, 7, 8]
# ciq.countQuadruplets(nums)
# from file_processing import FileSystem, Node

# fileSystem = FileSystem()
# input1 = ["FileSystem", "ls", "mkdir", "addContentToFile", "ls", "readContentFromFile"]
# input2 = [[], ["/"], ["/a/b/c"], ["/a/b/c/d", "hello"], ["/"], ["/a/b/c/d"]]
# fileSystem.mkdir("/a/b/c")
# fileSystem.addContentToFile("/a/b/c/d", "hello");
# fileSystem.ls("/");
# fileSystem.readContentFromFile("/a/b/c/d");
# from SlideWindows.counts import FreqMostFreqElements
# fmfe = FreqMostFreqElements()
# nums = [9953,9960,9908,9957,9919,9967,9941,9985,9925,9933,9989,9999,9928,9990,9973,9930,9982,9911,9986,9931,9925,9943,9937,9956,9968,9988,9929,9997,9945,9931,9922,9948,9916,9948,9998,9967,9945,9906,9914,9947,9997,9945,9923,9969,9903,9947,9938,9972,9969,9953,9926,9949,9997,9971,9913,9948,9910,9964,9900,9983,9945,9900,9951,9928,9984,9960,9903,9903,9983,9920,9909,9927,9987,9994,9987,9965,9941,9921,9914,9936,9979,9917,9965,9906,9942,9904,9920,9907,9922,9983,9970,9963,9941,9902,9968,9992,9994,9954,9904,9974,9914,9903,9934,10000,9991,9991,9986,9965,9980,9907,9911,9918,9993,9981,9986,9986,9944,9973,9918,9931,9974,9976,9958,9987,9942,9995,9970,9963,9901,9979,9995,9936,9959,9965,9905,9979,9927,9989,9926,9984,9956,9936,9931,9954,9901,9949,9943,9945,9966,9973,9931,9970,9916,9981,9995,9981,9968,9942,9960,10000,9935,9957,9931,9964,9939,9979,9924,9973,9960,9972,9915,9981,9993,9961,9963,9970,9917,9955,9993,9930,9972,9940,9921,9978,9915,9988,9904,9989,9911,9958,9914,9901,9913,9916,9909,9926,9928,9926,9920,9958,9931,9906,9973,9960,9909,9948,9983,9948,9936,9953,9974,9940]
# fmfe.maxFrequency(nums, 410)
# from StringProblems.string_comparisons import StringProblemClass
# spc = StringProblemClass()
# strs = ["abab","aba","abc"]
#spc.longestCommonPrefix_efficient(strs)
# from BackTrack.combination_permutation_problems import CombinationSumSeries
# css = CombinationSumSeries()
# candidates = [10,1,2,7,6,1,5]
# target = 8
# css.combinationSum2(candidates, target)
# from math_related import count_trailing_zeros, find_pythagorean_triplets
# print(count_trailing_zeros(100))
# # Example usage
# arr = [3, 1, 4, 6, 5]
# print(find_pythagorean_triplets(arr))  # Should return True for this example


# from DynamicProgramming.dynamic_programming import CoinChange
# cc = CoinChange()
# coins = [1,2,5]
# amount = 11
# print(cc.coinChange(coins, amount))
# from GraphProblems.valid_graph import UnionFind, GraphValidTree
# n = 5
# edges = [[0,1],[1,2],[2,3],[1,3],[1,4]]
# gvt = GraphValidTree()
# gvt.validTree(n, edges)
# n = 5
# edges = [[0,1],[0,2],[0,3],[1,4]]
# gvt.validTree(n, edges)

# from StringProblems.string_problems import StringProblemsClass
# spc = StringProblemsClass()
# #s = "AABABBA"
# #s = 'AAAA'
# s = "ABCDE"
# print(spc.characterReplacement(s, 1))

# from math_related import ValidNumber
# vn = ValidNumber()
# s="2e0"
#print(vn.isNumber(s))
# from DynamicProgramming.dynamic_programming import TargetSum
# nums = [1,1,1,1,1]
# target = 3
# ts = TargetSum()
# ts.findTargetSumWays_dp(nums, target)

# from binary_search_related import SplitArrayLargestSum
# nums = [7,2,5,10,8,11,4,9]
# k = 3
# sls = SplitArrayLargestSum()
# sls.splitArray(nums, k)
# from array_problems.array_operations import DiagonalTravasal
# dt = DiagonalTravasal()
# mat = [[1,2,3],[4,5,6],[7,8,9]]
# Output: [1,2,4,7,5,3,6,8,9]
# print(dt.findDiagonalOrder(mat))

# from array_problems.sum_series import Continuous_Subarray_Sum
# css = Continuous_Subarray_Sum()
# nums = [23,2,6,4,7]
# k = 6
# print(css.checkSubarraySum(nums, 6))
# from SortingProblems.sorting_related import MinMoves
# mm = MinMoves()
# grid = [[1,0,0,0,1],[0,0,0,0,0],[0,0,1,0,0]]
# print(mm.minTotalDistance_v2(grid))
# from GraphProblems.graph import GraphRelatedProblems
# # Test
# matrix = [
#   [0, 0, 1, 0],
#   [1, 0, 1, 0],
#   [1, 0, 0, 0],
#   [0, 1, 1, 0]
# ]
# grp = GraphRelatedProblems()
# print(grp.find_path(matrix))
# matrix = [
#     [1, 3, 1],
#     [1, 5, 1],
#     [4, 2, 1]
# ]
# print(grp.minMaxPath(matrix)) ## Dijkstra
# from SlideWindows.string_dp import LongestSubstrNoRepchar
# from SlideWindows.string_dp import SuperArraySubArray
# sasa = SuperArraySubArray()
# # Test
# k = 3
# arr = [1, 2, 2, 2, 1, 1, 1, 1]
# print(sasa.super_array_subarray(k, arr))
# from array_problems.sub_array_problems import SparseVector, SparseVectorBinarySearch
# nums1 = [0,1,0,0,2,0,0]
# nums2 = [1,0,0,0,3,0,4]
# sv1 = SparseVectorBinarySearch(nums=nums1)
# sv2 = SparseVectorBinarySearch(nums=nums2)
# sv1.dotProduct(sv2)
# sv1 = SparseVector(nums=nums1)
# sv2 = SparseVector(nums=nums2)
# sv1.dotProduct_binarysearch(sv2)

# from StringProblems.string_problems import StringProblemsClass
# spc = StringProblemsClass()
# num1 = "123"
# num2 = "456"
# spc.multiply(num1, num2)
# from backtrack import DiverseBackTrack, ProductOfPrimes
# dbt = DiverseBackTrack()
# primes = [2, 3, 11]
# dbt.print_products(primes)
# # Test
# primes = [2, 3, 11]
# obj = ProductOfPrimes()
# print(obj.get_product(primes))
# from HashMap.hash_map_related import HashMapRelated
# hmr = HashMapRelated()
# nums = [1,-1,0]
# k = 0
# hmr.subarraySum(nums, k)
# from StringProblems.string_problems import StringProblemsClass
#
# spc = StringProblemsClass()
# # 示例
# input_str = "helluoworldz"
# order = "eholwrd"
# print(spc.customSort(input_str, order))  # 输出 "ehollowrdz"
#
# # 示例
# input_str = "helloworld"
# order = "eholwrd"
# print(spc.customSort(input_str, order))  # 输出 "ehollowrd"
# s = "the sky is blue"
# print(spc.reverseWords(s))
# from Trees.binary_tree import TreeNode
# tree_node = TreeNode(-10)
# tree_node.left = TreeNode(9)
# tree_node.right = TreeNode(20)
# tree_node.right.left = TreeNode(15)
# tree_node.right.right = TreeNode(7)
# from Trees.binary_tree import BinaryTreeComputation
# btc = BinaryTreeComputation()
# print(btc.maxPathSum_iter(tree_node))
# for A in range(1, 10):
#     for B in range(1, 10):
#         if 100*A*B + 10*A**2 + 49*B + 6*A == 2900:
#             print(f"A = {A}, B = {B}")
#             break
# from array_problems.sub_array_problems import ShortestSubArrMaxsum
# from StringProblems.string_comparisons import StringProblemClass
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

# words = ["wrt","wrf","er","ett","rftt"]
# # words = ["z","x","z"]
# from topological_sorting import TopologicalSorting
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

