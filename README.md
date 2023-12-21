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
12. 






