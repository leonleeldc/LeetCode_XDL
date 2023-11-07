from typing import List
'''
338. Counting Bits
Given an integer n, return an array ans of length n + 1 such that for each i (0 <= i <= n), ans[i] is the number of 1's in the binary representation of i.
Example 1:
Input: n = 2
Output: [0,1,1]
Explanation:
0 --> 0
1 --> 1
2 --> 10
Example 2:

Input: n = 5
Output: [0,1,1,2,1,2]
Explanation:
0 --> 0
1 --> 1
2 --> 10
3 --> 11
4 --> 100
5 --> 101
'''
class CountingBits:
  def countBitsDP(self, n: int) -> List[int]:
    ans = [0] * (n + 1)
    for i in range(1, n + 1):
      ans[i] = ans[i & (i - 1)] + 1
    return ans
'''
191. Number of 1 Bits
Write a function that takes the binary representation of an unsigned integer and returns the number of '1' bits it has (also known as the Hamming weight).

Note:

Note that in some languages, such as Java, there is no unsigned integer type. In this case, the input will be given as a signed integer type. It should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 3, the input represents the signed integer. -3.


Example 1:

Input: n = 00000000000000000000000000001011
Output: 3
Explanation: The input binary string 00000000000000000000000000001011 has a total of three '1' bits.
Example 2:

Input: n = 00000000000000000000000010000000
Output: 1
Explanation: The input binary string 00000000000000000000000010000000 has a total of one '1' bit.
Example 3:

Input: n = 11111111111111111111111111111101
Output: 31
Explanation: The input binary string 11111111111111111111111111111101 has a total of thirty one '1' bits.
'''
class NumberOneBits:
  def hammingWeight(self, n: int) -> int:
    c1 = 0
    count = 0
    leng = len(bin(n)) - 2  # bin() includes '0b' prefix, so subtract 2
    while count < leng:
      if (n & 1) == 1:
        c1 += 1
      n = n >> 1
      count += 1
    return c1


'''
190. Reverse Bits
Reverse bits of a given 32 bits unsigned integer.
Example 1:

Input: n = 00000010100101000001111010011100
Output:    964176192 (00111001011110000010100101000000)
Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.
Example 2:

Input: n = 11111111111111111111111111111101
Output:   3221225471 (10111111111111111111111111111111)
Explanation: The input binary string 11111111
'''
class ReverseBits:
  def reverseBits(self, n: int) -> int:
    bitMask = 0x80000000
    reverseBitTester = 1
    result = 0
    for x in range(32):
      if (reverseBitTester & n) != 0:
        result = result | bitMask
      bitMask = (bitMask >> 1) & 0x7FFFFFFF  # Logical right shift by 1
      reverseBitTester = reverseBitTester << 1
    return result
