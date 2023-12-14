# EditCodes

codes from the leetcode are stored here

2351. First Letter to Appear Twice
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
