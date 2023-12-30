# EditCodes

codes from the leetcode are stored here


1. (2351) First Letter to Appear Twice
Ans - class Solution(object):
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
2. (1)Two Sum
Ans class Solution:
    def twoSum(self, nums, target):
        num_indices = {}
        
        for i, num in enumerate(nums):
            complement = target - num
            if complement in num_indices:
                return [num_indices[complement], i]
            num_indices[num] = i
    
3. (1929) Concatenation of Array
Ans  class Solution:
    def getConcatenation(self, nums: List[int]) -> List[int]:
        ans = nums + nums
        return ans

4. (1920) Build Array from Permutation
Ans class Solution:
    def buildArray(self, nums: List[int]) -> List[int]:

        n = len(nums)
        ans = [0] * n
        for i in range(n):
            ans[i] = nums[nums[i]]
        return ans

5. (1512.) Number of Good Pairs   (  A good pair is defined as a pair of indices (i, j) where nums[i] is equal to nums[j] and i is less than j.)
Ans class Solution:
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

6. (2011.) Final Value of Variable After Performing Operations
Ans class Solution:
    def finalValueAfterOperations(self, operations: List[str]) -> int:
        X = 0
        for operation in operations:
            if operation == '++X' or operation == 'X++':
                X += 1
            if operation == '--X' or operation == 'X--':
                X -= 1
        return X


7. (1470.) Shuffle the Array
Ans  class Solution:
    def shuffle(self, nums: List[int], n: int) -> List[int]:
        result = []
        for i in range(n):
            result.append(nums[i])
            result.append(nums[i+n])
        return result


8. (2942.) Find Words Containing Character 
Ans class Solution:
    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        result = []
        for index,  word in enumerate(words):
            if x in word:
                result.append(index)
        return result
9. (1672.) Richest Customer Wealth
Ans class Solution:
    def maximumWealth(self, accounts: List[List[int]]) -> int:
        max_wealth = 0
        for customer in accounts:
            wealth = sum(customer)

            if wealth > max_wealth:
                max_wealth = wealth
        return max_wealth
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
15. (704.) Binary Search
    class Solution:
    def search(self, nums: List[int], target: int) -> int:
        left , right = 0, len(nums) - 1

        while left < right:
            mid = left + right - left // 2

            if nums[mid] == target:
                return mid
            elif nums[mid] < target :
                left = mid + 1
            elif nums[mid] > target:
                right = mid - 1
        return -1
16. 

    
