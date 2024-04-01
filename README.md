# EditCodes

codes from the leetcode are stored here


1. [(2351) First Letter to Appear Twice](https://leetcode.com/problems/first-letter-to-appear-twice/)
Ans: 
<script src="https://cdnjs.cloudflare.com/ajax/libs/clipboard.js/2.0.8/clipboard.min.js"></script>
<pre>
<code id="code-block" class="language-python">
class Solution(object):
    def repeatedCharacter(self, s):
        seen = {}
        first_repeated = ''
        for i, char in enumerate(s):
            if char in seen and seen[char][1] is None:
                seen[char][1] = i
                if first_repeated == '' or i < seen[first_repeated][1]:
                    first_repeated = char
            else:
                seen[char] = [i, None]
        return first_repeated
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

<script>
  var clipboard = new ClipboardJS('.btn');

  clipboard.on('success', function(e) {
    console.log('Copied!');
    e.clearSelection();
  });
</script>

2. [(1)Two Sum](https://leetcode.com/problems/two-sum/)
   
Ans:
<pre>
<code id="code-block" class="language-python">
class Solution:
    def twoSum(self, nums, target):
        num_indices = {}
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_indices:
                return [num_indices[complement], i]
            num_indices[num] = i
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>
    
3. [(1929) Concatenation of Array](https://leetcode.com/problems/concatenation-of-array/)

Ans:
<pre>
<code id="code-block" class="language-python">
class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        ans = nums + nums
        return ans
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

4. [(1920) Build Array from Permutation](https://leetcode.com/problems/build-array-from-permutation/)

Ans:
<pre>
<code id="code-block" class="language-python">
class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:
        n = len(nums)
        ans = [0] * n
        for i in range(n):
            ans[i] = nums[nums[i]]
        return ans
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

5. [(1512.) Number of Good Pairs](https://leetcode.com/problems/number-of-good-pairs/)   (  A good pair is defined as a pair of indices (i, j) where nums[i] is equal to nums[j] and i is less than j.)
Ans
<pre>
<code id="code-block" class="language-python">
 class Solution:
    def numIdenticalPairs(self, nums: List[int]) -> int:
        count = 0
        num_counts = {}
        for num in nums:
            if num in num_counts:
                num_counts[num] += 1
            else:
                num_counts[num] = 1
        for key in num_counts:
            occurence = num_counts[key]
            if occurence > 1:
                count += (occurence * (occurence - 1)) // 2       # formula used is nc2 to identify the occurences
        return count
    </code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

6. [(2011.) Final Value of Variable After Performing Operations](https://leetcode.com/problems/final-value-of-variable-after-performing-operations/)

Ans 
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        X = 0
        for operation in operations:
            if operation == '++X' or operation == 'X++':
                X += 1
            if operation == '--X' or operation == 'X--':
                X -= 1
        return X
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>


7. [(1470.) Shuffle the Array](https://leetcode.com/problems/shuffle-the-array/)

Ans
<pre>
<code id="code-block" class="language-python">
class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        result = []
        for i in range(n):
            result.append(nums[i])
            result.append(nums[i+n])
        return result
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

8. [(2942.) Find Words Containing Character ](https://leetcode.com/problems/find-words-containing-character/)

Ans 
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        result = []
        for index,  word in enumerate(words):
            if x in word:
                result.append(index)
        return result
    </code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

9. [(1672.) Richest Customer Wealth](https://leetcode.com/problems/richest-customer-wealth/)

Ans 
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        max_wealth = 0
        for customer in accounts:
            wealth = sum(customer)
            if wealth > max_wealth:
                max_wealth = wealth
        return max_wealth
    </code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

10. (2798.) Number of Employees Who Met the Target

Ans class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        count = 0
        for hour in hours:
            if hour >= target: 
                count += 1
        return count
          or
    class Solution:
    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        return sum(1 for hour in hours if hour >= target)

11. (1431.) Kids With the Greatest Number of Candies

Ans python class Solution(object):
    def kidsWithCandies(self, candies, extraCandies):
        max_candies = max(candies)
        result = []
        for candy in candies:
            if candy + extraCandies >= max_candies:
                result.append(True)
            else:
                result.append(False)
        return result 
    python3 class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        result = []
        max_candies = max(candies)
        for candy in candies:
            if candy + extraCandies >= max_candies:
                result.append(True)
            else:
                result.append(False)
        return result

12. (136.) Single Number

Ans  class Solution:
    def singleNumber(self, nums: List[int]) -> int:
        result = 0
        for num in nums:
            result ^= num     # we xor each element in the row.  
        return result

'''since in xor same element result in 0 and different element result in 1. it takes the binary value and then it does the xor operation. 
# For nums = [4, 1, 2, 1, 2]:
 result starts as 0.
result ^= 4 -> result = 0 ^ 4 = 4.
result ^= 1 -> result = 4 ^ 1 = 5.
result ^= 2 -> result = 5 ^ 2 = 7.
result ^= 1 -> result = 7 ^ 1 = 6.
result ^= 2 -> result = 6 ^ 2 = 4. '''

13. (169.) Majority Element

Ans class Solution:
    def majorityElement(self, nums: List[int]) -> int:
        count = 0
        canditate = None
        for num in nums:
            if count == 0:
                candidate = num
            count += 1 if num == candidate else -1    # if the count is <1 after the loop in the entire array then the candidate takes the next element as the candidate and the process continues until the count is > 1  
        return candidate
''' 
The above code selects the first element and then loops around the entire array if the element in the index is the same as the candidate one then the count is increased. the logic of the thing is that the count of the majority element will not be 0 
'''

14. (283.) Move Zeroes

    class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        non_zero_index = 0
        """
        Do not return anything, modify nums in place instead.
        """
        for i in range(len(nums)):
            if nums[i] != 0 :   # check the current value is non zero
                nums[non_zero_index], nums[i] = nums[i] , nums[non_zero_index] 
                non_zero_index += 1  # if non zero then swap with non-zero indexes which are 0 and then increase by one value  ie move on to the next index.
        while non_zero_index < len(nums):
            nums[non_zero_index] = 0   # after all the non-zero values the remaining will be zeroes this ensures that
            non_zero_index += 1  # Move on to the next index
16. (704.) Binary Search

    class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left , right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target :
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
        return -1 
17. (121.) Best Time to Buy and Sell Stock

     class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        if not prices or len(prices) == 1:
            return 0   
        min_prices = prices[0]
        max_profit = 0
        for price in prices[1:]:
            if price < min_prices :
                min_prices = pric
            else:
                max_profit = max(max_profit, price - min_prices)     
        return max_profit

19.  (35.) Search Insert Position
    class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        left , right = 0, len(nums) - 1
        while left <= right:
            mid = left + (right - left) // 2
            if target == nums[mid]:
                return mid
            if target < nums[mid]:
                right = mid -1
            if target > nums[mid]:
                left = mid +1
        return left  
another code is that

from bisect import bisect_left
class Solution:
    def searchInsert(self, nums: List[int], target: int) -> int:
        return bisect_left(nums,target)

20. (118.) Pascal's Triangle
Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it 
Ans 
class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        if numRows <= 0:
            return []
        res = [[1]]
        for i in range(numRows - 1):
            temp = [0] + res[-1] + [0]
            row = []
            for j in range(len(res[-1]) + 1):
                row.append(temp[j] + temp[j+1])
            res.append(row)
        return res
        
21. (46.) Permutations 
Given an array nums of distinct integers, return all the possible permutations. You can return the answer in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]
Example 2:

Input: nums = [0,1]
Output: [[0,1],[1,0]]
Example 3:

Input: nums = [1]
Output: [[1]]

Ans: class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        def backtrack(start):
            if start == len(nums):
                result.append(nums.copy())
                return

            for i in range(start , len(nums)):
                nums[start], nums[i] = nums[i], nums[start]
                backtrack(start + 1)
                nums[start] , nums[i] = nums[i], nums[start]
        
        result = []
        backtrack(0)
        return result

22. (78) 78. Subsets
Given an integer array nums of unique elements, return all possible 
subsets(the power set).
The solution set must not contain duplicate subsets. Return the solution in any order.

Example 1:

Input: nums = [1,2,3]
Output: [[],[1],[2],[1,2],[3],[1,3],[2,3],[1,2,3]]
Example 2:

Input: nums = [0]
Output: [[],[0]]
 

Constraints:

1 <= nums.length <= 10
-10 <= nums[i] <= 10
All the numbers of nums are unique.

Ans: from itertools import combinations
from typing import List

class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        final_result = []
        for n in range(len(nums) + 1):
            pairs = [list(pair) for pair in combinations(nums, n)]
            final_result += pairs
        return final_result
23.(48) 48. Rotate Image
You are given an n x n 2D matrix representing an image, rotate the image by 90 degrees (clockwise).
You have to rotate the image in place, which means you have to modify the input 2D matrix directly. DO NOT allocate another 2D matrix and do the rotation.

Example 1:
Input: matrix = [[1,2,3],[4,5,6],[7,8,9]]
Output: [[7,4,1],[8,5,2],[9,6,3]]
Example 2:
Input: matrix = [[5,1,9,11],[2,4,8,10],[13,3,6,7],[15,14,12,16]]
Output: [[15,13,2,5],[14,3,4,1],[12,6,8,9],[16,7,10,11]]
Constraints:

n == matrix.length == matrix[i].length
1 <= n <= 20
-1000 <= matrix[i][j] <= 1000

Ans class Solution:
    def rotate(self, matrix: List[List[int]]) -> None:
        n = len(matrix)
        for i in range(n):
            for j in range(i , n):
                matrix[i][j],matrix[j][i] = matrix[j][i],matrix[i][j]
        for row in matrix:
            row.reverse()

24.(39) Combination Sum
Given an array of distinct integer candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to the target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the 
frequency
 of at least one of the chosen numbers is different.

The test cases are generated such that the number of unique combinations that sum up to the target is less than 150 combinations for the given input.
 

Example 1:
Input: candidates = [2,3,6,7], target = 7
Output: [[2,2,3],[7]]
Explanation:
2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
7 is a candidate, and 7 = 7.
These are the only two combinations.
Example 2:
Input: candidates = [2,3,5], target = 8
Output: [[2,2,2,2],[2,3,3],[3,5]]
Example 3:
Input: candidates = [2], target = 1
Output: []
 
Constraints:

1 <= candidates.length <= 30
2 <= candidates[i] <= 40
All elements of candidates are distinct.
1 <= target <= 40

Ans: class Solution:
    def combinationSum(self, candidates: List[int], target: int) -> List[List[int]]:
        candidates.sort()
        result = []

        def backtrack(start, target , combination):
            if target == 0:
                result.append(combination[:])
                return

            for i in range(start , len(candidates)):
                if candidates[i] > target:
                    break
                combination.append(candidates[i])
                backtrack(i , target - candidates[i], combination)
                combination.pop()
        backtrack(0 , target, [])
        return result

25. (1952).  Three Divisors
Hint
Given an integer n, return true if n has exactly three positive divisors. Otherwise, return false.

An integer m is a divisor of n if there exists an integer k such that n = k * m.
Example 1:

Input: n = 2
Output: false
Explantion: 2 has only two divisors: 1 and 2.
Example 2:

Input: n = 4
Output: true
Explantion: 4 has three divisors: 1, 2, and 4.
Constraints:

1 <= n <= 104
Ans : class Solution:
    def isThree(self, n: int) -> bool:
        if n <= 3:
            return False
        count = 0
        for i in range(2, n//2 + 1):
            if n % i == 0:
                count += 1
            if count > 1:
                return False
        if count == 0:
            return False
        return True

26. 2427. Number of Common Factors

Given two positive integers a and b, return the number of common factors of a and b.

An integer x is a common factor of a and b if x divides both a and b. 

Example 1:

Input: a = 12, b = 6
Output: 4
Explanation: The common factors of 12 and 6 are 1, 2, 3, 6.
Example 2:

Input: a = 25, b = 30
Output: 2
Explanation: The common factors of 25 and 30 are 1, 5.
 
Constraints:

1 <= a, b <= 1000
Ans: class Solution:
    def commonFactors(self, a: int, b: int) -> int:
        x = min(a, b) + 1
        com = []
        for i in range(1, x):
            if a % i == 0 and b % i == 0 :
                com.append(i)
        return len(com)

27. [1979. Find Greatest Common Divisor of Array](https://leetcode.com/problems/find-greatest-common-divisor-of-array/description/)
Ans: class Solution:
    def findGCD(self, nums: List[int]) -> int:
        x = min(nums)
        y = max(nums)
        div = []
        for i in range(1, x+1 ):
            if x % i == 0 and y % i == 0:
                div.append(i)
        return max(div)
28. [1108. Defanging an IP Address](https://leetcode.com/problems/defanging-an-ip-address/description/)
Ans: class Solution:
    def defangIPaddr(self, address: str) -> str:
        new_address = address.replace(".", "[.]")
        return new_address

29. [1. Two Sum](https://leetcode.com/problems/two-sum/description/)
Ans: class Solution(object):
    def twoSum(self, nums, target):
        n = len(nums)
        for i in range(n):
            for j in range(i+1, n):
                if nums[i] + nums[j] == target:
                    return [i,j]
        return []
30. [412. Fizz Buzz](https://leetcode.com/problems/fizz-buzz/description/)
Ans: class Solution:
    def fizzBuzz(self, n: int) -> List[str]:
        string = []
        for i in range(1,n+1):
            if i % 3 == 0 and i % 5 == 0:
                string.append("FizzBuzz")
            elif i % 5 == 0:
                string.append("Buzz")
            elif i % 3 == 0:
                string.append("Fizz")
            else:
                string.append(str(i))
        return string
31. [709. To Lower Case](https://leetcode.com/problems/to-lower-case/description/)
Ans: class Solution:
    def toLowerCase(self, s: str) -> str:
        return s.lower()

32. [2833. Furthest Point From Origin](https://leetcode.com/problems/furthest-point-from-origin/description/)
Ans: class Solution:
    def furthestDistanceFromOrigin(self, moves: str) -> int:
        return moves.count('_') + abs(moves.count('R') - moves.count('L'))

33. [657. Robot Return to Origin](https://leetcode.com/problems/robot-return-to-origin/)
Ans: class Solution:
    def judgeCircle(self, moves: str) -> bool:
        ud , lr = 0, 0 
        for move in moves:
            if move == 'U':
                ud += 1
            elif move == 'D':
                ud -= 1
            elif move == 'R':
                lr += 1
            else:
                lr -= 1
        if ud == 0 and lr == 0:
            return True
34. [1281. Subtract the Product and Sum of Digits of an Integer](https://leetcode.com/problems/subtract-the-product-and-sum-of-digits-of-an-integer/)
Ans: class Solution:
    def subtractProductAndSum(self, n: int) -> int:
        product = 1
        sum = 0
        num_str = str(n)
        num_list = [int(digit) for digit in num_str]
        for i in num_list:
            product *= i
            sum += i
        return product - sum
35. [2535. Difference Between Element Sum and Digit Sum of an Array](https://leetcode.com/problems/difference-between-element-sum-and-digit-sum-of-an-array/)
Ans: class Solution:
    def differenceOfSum(self, nums: List[int]) -> int:
        element_sum , digit_sum  = 0 , 0
        digits = [int(digit) for num in nums for digit in str(num)]
        for i in nums:
            element_sum += i
        for digit in digits:
            digit_sum += digit
        return element_sum - digit_sum

36. [2180. Count Integers With Even Digit Sum](https://leetcode.com/problems/count-integers-with-even-digit-sum/)
Ans: class Solution:
    def countEven(self, num: int) -> int:
        count = 0
        for i in range(1, num + 1):
            digit_sum = sum([int(digit) for digit in str(i)])
            if digit_sum % 2 == 0:
                count += 1
        return count
37. [507. Perfect Number](https://leetcode.com/problems/perfect-number/)
Ans: class Solution:
    def checkPerfectNumber(self, num: int) -> bool:
        if num <= 1:
            return False
        pos = 1
        for i in range(2, int(sqrt(num))+ 1):
            if num % i == 0:
                pos += i
                if i != num // i:          #the range will be upto the square root.
                    pos += num // i         # To add divisors > than that we add the counter parts of the divisible number
        if pos == num:
            return True
        else:
            return False

38. [1816. Truncate Sentence](https://leetcode.com/problems/truncate-sentence/)
Ans:
<pre>
<code id="code-block" class="language-python">
class Solution:
    def truncateSentence(self, s: str, k: int) -> str:
        s = s.split()
        s = s[:k]
        s = ' '.join(s)
        return s
    </code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

    

39. [557. Reverse Words in a String III](https://leetcode.com/problems/reverse-words-in-a-string-iii/)

<pre>
<code id="code-block" class="language-python">
class Solution:
    def reverseWords(self, s: str) -> str:
        reversed_string = [word[::-1] for word in s.split()]
        r_s = ' '.join(reversed_string)
        return r_s
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

40. [2706. Buy Two Chocolates](https://leetcode.com/problems/buy-two-chocolates/)
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def buyChoco(self, prices: List[int], money: int) -> int:
        # Sort the Array in Increasing Order
        prices.sort()

        # Minimum Cost
        min_cost = prices[0] + prices[1]

        # We can buy chocolates only if we have enough money
        if min_cost <= money:
            # Return the Amount of Money Left
            return money - min_cost
        else:
            # We cannot buy chocolates. Return the initial amount of money
            return money

</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

41. [2894. Divisible and Non-divisible Sums Difference](https://leetcode.com/problems/divisible-and-non-divisible-sums-difference/)
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def differenceOfSums(self, n: int, m: int) -> int:
        num1 , num2 = 0, 0
        
        for i in range(1, n+1):
            if i % m == 0:
                num2 += i
            else:
                num1 += i

        return num1 - num2
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

42. [520. Detect Capital](https://leetcode.com/problems/detect-capital/)
<pre>
<code id="code-block" class="language-python">
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if len(word) == 0:
            return True
        
        first_capital = word[0].isupper() and all(i.islower() for i in word[1:])
        all_upper = all(i.isupper() for i in word)
        all_lower = all(i.islower() for i in word)

        return first_capital or all_upper or all_lower
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

<pre>
<code id="code-block" class="language-python">
class Solution:
    def detectCapitalUse(self, word: str) -> bool:
        if word.upper() == word:
            return True
        elif word.lower() == word:
            return True
        elif word.capitalize() == word:
            return True
        return False
        return first_capital or all_upper or all_lower
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

43. [1662. Check If Two String Arrays are Equivalent](https://leetcode.com/problems/check-if-two-string-arrays-are-equivalent/)
<pre>
<code id="code-block" class="language-python">
class Solution:
    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        l1 = ''.join(word1)
        l2 = ''.join(word2)

        if l1 == l2:
            return True
        else:
            return False
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

44. [1528. Shuffle String](https://leetcode.com/problems/shuffle-string/)
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def restoreString(self, s: str, indices: List[int]) -> str:
        combined_list = list(zip(indices, s))
        combined_list.sort()
        sorted_list = ''.join(letter for _,letter in combined_list) # _ , is just used as an placeholder 
        return sorted_list                                   # It does not take the value of number it skips it. and only take the string part


</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

45. [2154. Keep Multiplying Found Values by Two](https://leetcode.com/problems/keep-multiplying-found-values-by-two/)
<pre>
<code id="code-block" class="language-python">
class Solution:
    def findFinalValue(self, nums: List[int], original: int) -> int:
        while original in nums:
            original *= 2
        return original
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

46. [191. Number of 1 Bits](https://leetcode.com/problems/number-of-1-bits/)
<pre>
<code id="code-block" class="language-python">
class Solution:
    def hammingWeight(self, n: int) -> int:
        count = 0
        while n:
            count += n & 1
            n >>= 1
        return count
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

47. [509. Fibonacci Number](https://leetcode.com/problems/fibonacci-number/)
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def fib(self, n: int) -> int:
        f1 , f2 = 0,  1

        for i in range(n):
            f1 , f2 = f2 , f1 + f2
        return f1
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

48. [1678. Goal Parser Interpretation](https://leetcode.com/problems/goal-parser-interpretation/)
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def interpret(self, command: str) -> str:
        com = ''
        i = 0
        while i < len(command):
            if command[i] == 'G':
                com += 'G'
                i = i + 1
            elif command[i: i +2 ] == '()':
                com += 'o'
                i = i + 2
            elif command[i : i+ 4] == '(al)':
                com += 'al'
                i = i + 4
        return com
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>

49. [2769. Find the Maximum Achievable Number](https://leetcode.com/problems/find-the-maximum-achievable-number/)
<pre>
<code id="code-block" class="language-python">
class Solution:
    def theMaximumAchievableX(self, num: int, t: int) -> int:
         return num+t+t
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  Copy code
</button>
        
50 [268. Missing Number](https://leetcode.com/problems/missing-number/)
<pre>
<code id="code-block" class="language-python">
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        n = len(nums)
        expected_sum = (n * (n+1)) // 2
        actual_sum = sum(nums)
        return expected_sum - actual_sum
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">

</button>      

51. [2544. Alternating Digit Sum](https://leetcode.com/problems/alternating-digit-sum/)
<pre>
<code id="code-block" class="language-python">
    class Solution:
    def alternateDigitSum(self, n: int) -> int:
        n_str = str(n)
        num1 = [int(n_str[i]) for i in range(len(n_str)) if i % 2 != 0]    # here in string the right most digit has the index 0 that why the positive sign numbers go to num2
        num2 = [int(n_str[i]) for i in range(len(n_str)) if i % 2 == 0]
        return sum(num2) - sum(num1)       
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  
</button>  
52. [2129. Capitalize the Title](https://leetcode.com/problems/capitalize-the-title/)
<pre>
<code id="code-block" class="language-python">
class Solution:
    def capitalizeTitle(self, title: str) -> str:
        word = title.split()
        cap_word = [i.capitalize() if len(i) >= 3 else i.lower() for i in word]
        cap_sen = " ".join(cap_word)
        return cap_sen
</code>
</pre>

<button class="btn" data-clipboard-target="#code-block">
  
</button>  
