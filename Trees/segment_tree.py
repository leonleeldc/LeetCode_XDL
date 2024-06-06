'''
All/Most of the segement tree operations are in post order traversal.
'''
class SegmentTree:
    def __init__(self, interviewers, interview_time):
        self.interviewers_counts = {name: 0 for name in interviewers}
        self.output = {}
        self.interview_time = interview_time
        tuple_inputs = []
        for interviewer in interviewers:
            time_ranges = interviewers[interviewer]
            for time_range in time_ranges:
                tuple_inputs.append([time_range[0], time_range[1], interviewer])
        tuple_inputs.sort(key=lambda x: x[0])
        self.root = self.createTree(tuple_inputs)

    # Create tree for range [i,j] from array nums
    def createTree(self, interviewers):
        def helper(i, j):
            # Base case
            if i > j:
                return None

            # Base case - Create leaf node
            if i == j:
                print(f'i={i} j={j} and val={interviewers[j]}')
                return Node(i, j, interviewers[i][0], interviewers[i][1], interviewers[i][2])
            # Find mid index and recursively create left and right children
            mid = (i + j) // 2
            left = helper(i, mid)
            right = helper(mid + 1, j)
            # Create and return the current node based on left and right children
            return Node(i, j, interviewers[i][0], interviewers[j][1], interviewers[i][2]+' '+interviewers[j][2], left, right)
        return helper(0, len(interviewers) - 1)

    def rangeMatch(self, interviewee):
        if interviewee[2] in self.output:
            return self.output[interviewee[2]]
        left, right = interviewee[0], interviewee[1]
        def helper(node):
            # Base case current interval falls fully inside query interal [left,right]
            if node.start_range >= left and node.end_range <= right and len(node.interviewer.split())==1:
                ##['john', 'jerry', [2, 3]]
                self.output[interviewee[2]] = [interviewee[2], node.interviewer, [node.start_range, node.start_range + self.interview_time]]
                return True
            elif node.start_range<=left and node.end_range>=right and len(node.interviewer.split())==1: ##this means that we reach the leaf node and we cannot go deeper
                self.output[interviewee[2]] = [interviewee[2], node.interviewer,
                                               [interviewee[0], interviewee[0] + self.interview_time]]
                return True
            elif node.start_range<=left and node.end_range>=left+self.interview_time and len(node.interviewer.split())==1:
                self.output[interviewee[2]] = [interviewee[2], node.interviewer,
                                               [interviewee[0], interviewee[0] + self.interview_time]]
                return True
            # Find mid range and recursively find rangeMatch
            mid = (node.start_range + node.end_range) // 2
            # If query is only on left side
            if right <= mid and node.left:
                helper(node.left)
            # If query is only on right side
            elif left > mid and node.right:
                helper(node.right)
            # Else go on both sides
            elif node.left and node.right:
                helper(node.left)
                helper(node.right)
        return helper(self.root)

class Node:
    def __init__(self, s, e, st_r, et_r, n, l=None, r=None):
        self.start = s
        self.end = e
        self.start_range = st_r
        self.end_range = et_r
        self.interviewer = n ## for leaf node, only one person, for intermedidate nodes or root, all persons are included
        self.left = l
        self.right = r

if __name__ == '__main__':
    ##["NumArray", "rangeMatch", "update", "rangeMatch"]

    ##Input1 = {'john': [[1, 3], [4, 8]], 'jerry': [[2, 3], [11, 12]], 'jimmy': [[7, 9], [12, 14]], 'tom': [[3, 5], [9, 11]]}
    Input1 = {'john': [[1, 3], [4, 8]], 'tom': [[3, 5], [9, 11]]}
    Input2 = {'jerry': [[2, 3], [11, 12]], 'jimmy': [[7, 9], [12, 14]], 'josh': [[1, 2]]}
    Interview_time = 1
    Output = [['john', 'jerry', [2, 3]], ['john', 'jimmy', [7, 8]], ['john', 'josh', [1, 2]]]
    # input = [[[1, 3, 5]], [0, 2], [1, 2], [0, 2]]
    # num = [1, 4, 5, 0, 2, 8, 9, 15, 20]
    segment_tree = SegmentTree(Input1, Interview_time)
    tuple_inputs = []
    for interviewee in Input2:
        time_ranges = Input2[interviewee]
        for time_range in time_ranges:
            tuple_inputs.append([time_range[0], time_range[1], interviewee])
    tuple_inputs.sort(key=lambda x: x[0])
    #segment_tree.update(4, 24)
    for interviewee in tuple_inputs:
        segment_tree.rangeMatch(interviewee)
    for key in segment_tree.output:
        print(segment_tree.output[key])
    print('done')