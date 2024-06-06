'''
373. Find K Pairs with Smallest Sums
'''
from typing import List, Tuple
import heapq
'''
472. Concatenated Words
Solved
Hard
Topics
Companies
Given an array of strings words (without duplicates), return all the concatenated words in the given list of words.

A concatenated word is defined as a string that is comprised entirely of at least two shorter words (not necessarily distinct) in the given array.
'''


class ConcatenatedWords:
  def findAllConcatenatedWordsInADict(self, words: List[str]) -> List[str]:
    word_length_heap = []
    for word in words:
      heapq.heappush(word_length_heap, (len(word), word))

    def backtrack(ind, word):
      if not word: return True
      for end in range(len(word) + 1):
        if word[:end] in word_set and backtrack(ind, word[end:]):
          return True
      return False
    # for i in range(len(words)):
    #   print(word_length_heap[i])
    ans = []
    word_set = set()
    for i in range(1, len(words)):
      if word_length_heap[i - 1][1] not in ans:
        word_set.add(word_length_heap[i - 1][1])
      print(f'i={i} and word_length_heap[i] = {word_length_heap[i-1]}')
      if backtrack(i, word_length_heap[i][1]):
        ans.append(word_length_heap[i][1])
    return ans

'''
295. Find Median from Data Stream
The median is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

For example, for arr = [2,3,4], the median is 3.
For example, for arr = [2,3], the median is (2 + 3) / 2 = 2.5.
Implement the MedianFinder class:

MedianFinder() initializes the MedianFinder object.
void addNum(int num) adds the integer num from the data stream to the data structure.
double findMedian() returns the median of all elements so far. Answers within 10-5 of the actual answer will be accepted.
Example 1:

Input
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]
Output
[null, null, null, 1.5, null, 2.0]

Explanation
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr[1, 2, 3]
medianFinder.findMedian(); // return 2.0

Constraints:

-105 <= num <= 105
There will be at least one element in the data structure before calling findMedian.
At most 5 * 104 calls will be made to addNum and findMedian.

Follow up:

If all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
If 99% of all integer numbers from the stream are in the range [0, 100], how would you optimize your solution?
'''

import heapq

class MedianFinder:

    def __init__(self):
        self.min_heap = []  # To store the larger half of the numbers
        self.max_heap = []  # To store the smaller half of the numbers (negative values)

    def addNum(self, num: int) -> None:
        if len(self.max_heap) == 0 or num <= -self.max_heap[0]:
            heapq.heappush(self.max_heap, -num)
        else:
            heapq.heappush(self.min_heap, num)

        # Balance the heaps if necessary
        if len(self.max_heap) > len(self.min_heap) + 1:
            heapq.heappush(self.min_heap, -heapq.heappop(self.max_heap))
        elif len(self.min_heap) > len(self.max_heap):
            heapq.heappush(self.max_heap, -heapq.heappop(self.min_heap))

    def findMedian(self) -> float:
        if len(self.max_heap) == len(self.min_heap):
            return (-self.max_heap[0] + self.min_heap[0]) / 2.0
        else:
            return -self.max_heap[0]

class HeapRelated:
  def kSmallestPairs(self, nums1: List[int], nums2: List[int], k: int) -> List[List[int]]:
    ans, heap = [], []  # initialize variables
    visited = set()  # to keep a check whether index i,j are already visited
    i, j = 0, 0
    visited.add((i, j))  # add the i,j index to the visited
    heapq.heappush(heap, (nums1[i] + nums2[j], i, j))  # push the sum(nums1[i],nums2[j]) and i,j into the heap
    while k > 0 and heap:  # keep the loop untill k element are found or all the element are travelled
      _, i, j = heapq.heappop(heap)  # pop the element from the heap and append it to the answer
      ans.append([nums1[i], nums2[j]])
      if (i + 1) < len(nums1) and (i + 1,
                                   j) not in visited:  # check if (i + 1) < len(nums1) and (i+1),j are not in the visited then add the sum of i+1,j and i+1,j to the heap and visited
        heapq.heappush(heap, (nums1[i + 1] + nums2[j], i + 1, j))
        visited.add((i + 1, j))
      if (j + 1) < len(nums2) and (i, j + 1) not in visited:  # similarly as above check for the (j + 1)
        heapq.heappush(heap, (nums1[i] + nums2[j + 1], i, j + 1))
        visited.add((i, j + 1))
      k -= 1  # decrement the k
    return ans

'''
1834. Single-Threaded CPU
You are given n tasks labeled from 0 to n - 1 represented by a 2D integer array tasks, where tasks[i] = [enqueueTimei, processingTimei] means that the i​​​​​​th​​​​ task will be available to process at enqueueTimei and will take processingTimei to finish processing.

You have a single-threaded CPU that can process at most one task at a time and will act in the following way:

If the CPU is idle and there are no available tasks to process, the CPU remains idle.
If the CPU is idle and there are available tasks, the CPU will choose the one with the shortest processing time. If multiple tasks have the same shortest processing time, it will choose the task with the smallest index.
Once a task is started, the CPU will process the entire task without stopping.
The CPU can finish a task then start a new one instantly.
Return the order in which the CPU will process the tasks.
Example 1:

Input: tasks = [[1,2],[2,4],[3,2],[4,1]]
Output: [0,2,3,1]
Explanation: The events go as follows: 
- At time = 1, task 0 is available to process. Available tasks = {0}.
- Also at time = 1, the idle CPU starts processing task 0. Available tasks = {}.
- At time = 2, task 1 is available to process. Available tasks = {1}.
- At time = 3, task 2 is available to process. Available tasks = {1, 2}.
- Also at time = 3, the CPU finishes task 0 and starts processing task 2 as it is the shortest. Available tasks = {1}.
- At time = 4, task 3 is available to process. Available tasks = {1, 3}.
- At time = 5, the CPU finishes task 2 and starts processing task 3 as it is the shortest. Available tasks = {1}.
- At time = 6, the CPU finishes task 3 and starts processing task 1. Available tasks = {}.
- At time = 10, the CPU finishes task 1 and becomes idle.
Example 2:

Input: tasks = [[7,10],[7,12],[7,5],[7,4],[7,2]]
Output: [4,3,2,0,1]
Explanation: The events go as follows:
- At time = 7, all the tasks become available. Available tasks = {0,1,2,3,4}.
- Also at time = 7, the idle CPU starts processing task 4. Available tasks = {0,1,2,3}.
- At time = 9, the CPU finishes task 4 and starts processing task 3. Available tasks = {0,1,2}.
- At time = 13, the CPU finishes task 3 and starts processing task 2. Available tasks = {0,1}.
- At time = 18, the CPU finishes task 2 and starts processing task 0. Available tasks = {1}.
- At time = 28, the CPU finishes task 0 and starts processing task 1. Available tasks = {}.
- At time = 40, the CPU finishes task 1 and becomes idle.

'''
class SingleThreadedCPU:
  def getOrder(self, tasks: List[List[int]]) -> List[int]:
    # Sort based on min task processing time or min task index.
    next_task: List[Tuple[int, int]] = []
    tasks_processing_order: List[int] = []

    # Store task enqueue time, processing time, index.
    sorted_tasks = [(enqueue, process, idx) for idx, (enqueue, process) in enumerate(tasks)]
    sorted_tasks.sort()

    curr_time = 0
    task_index = 0

    # Stop when no tasks are left in array and heap.
    while task_index < len(tasks) or next_task:
      if not next_task and curr_time < sorted_tasks[task_index][0]:
        # When the heap is empty, try updating curr_time to next task's enqueue time.
        curr_time = sorted_tasks[task_index][0]

      # Push all the tasks whose enqueueTime <= currtTime into the heap.
      while task_index < len(sorted_tasks) and curr_time >= sorted_tasks[task_index][0]:
        _, process_time, original_index = sorted_tasks[task_index]
        heapq.heappush(next_task, (process_time, original_index))
        task_index += 1

      process_time, index = heapq.heappop(next_task)

      # Complete this task and increment curr_time.
      curr_time += process_time
      tasks_processing_order.append(index)

    return tasks_processing_order