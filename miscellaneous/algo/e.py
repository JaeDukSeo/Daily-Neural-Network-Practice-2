import numpy as np,sys
import itertools

#A naive recursive implementation of 0-1 Knapsack Problem
# https://codereview.stackexchange.com/questions/20569/dynamic-programming-solution-to-knapsack-problem
def knapsack(items, maxweight):
    """
    Solve the knapsack problem by finding the most valuable
    subsequence of `items` subject that weighs no more than
    `maxweight`.

    `items` is a sequence of pairs `(value, weight)`, where `value` is
    a number and `weight` is a non-negative integer.

    `maxweight` is a non-negative integer.

    Return a pair whose first element is the sum of values in the most
    valuable subsequence, and whose second element is the subsequence.

    >>> items = [(4, 12), (2, 1), (6, 4), (1, 1), (2, 2)]
    >>> knapsack(items, 15)
    (11, [(2, 1), (6, 4), (1, 1), (2, 2)])
    """

    # Return the value of the most valuable subsequence of the first i
    # elements in items whose weights sum to no more than j.
    def bestvalue(i, j):
        if i == 0: return 0
        value, weight = items[i - 1]
        if weight > j:
            return bestvalue(i - 1, j)
        else:
            return max(bestvalue(i - 1, j),
                       bestvalue(i - 1, j - weight) + value)

    j = maxweight
    result = []
    for i in range(len(items), 0, -1):
        if bestvalue(i, j) != bestvalue(i - 1, j):
            result.append(i-1)
            j -= items[i - 1][1]
    result.reverse()
    return bestvalue(len(items), maxweight), result

def partion(array):
    if sum(array)%2 == 0:
        temp = np.vstack((array,array)).T
        one = knapsack(temp,sum(array)/2)[1]
        two = [x for x in range(len(array)) if not x in one ]
        return [array[x] for x in one],[array[x] for x in two]
    else:
        return None



testing = [1,5,6,7,2,4,1]
print("Original Array : ",testing)
if sum(testing)%2==0: print('Sum Value: ', sum(testing)/2)
print(partion(testing))
print('-------------')

testing = [1,5,6,7,2,4]
print("Original Array : ",testing)
if sum(testing)%2==0: print('Sum Value: ', sum(testing)/2)
print(partion(testing))
print('-------------')

testing = [1,5,11,5,0,0,0,0]
print("Original Array : ",testing)
if sum(testing)%2==0: print('Sum Value: ', sum(testing)/2)
print(partion(testing))
print('-------------')

testing = [1,5,11,4]
print("Original Array : ",testing)
if sum(testing)%2==0: print('Sum Value: ', sum(testing)/2)
print(partion(testing))
print('-------------')

testing = [5,3,1,2,9,5,0,7]
print("Original Array : ",testing)
if sum(testing)%2==0: print('Sum Value: ', sum(testing)/2)
print(partion(testing))
print('-------------')


sys.exit()
# ------------------------------------------------------------
def isSubsetSum (arr, n, sum):
    # Base Cases
    if sum == 0:
        return True
    if n == 0 and sum != 0:
        return False
 
    # If last element is greater than sum, then 
    # ignore it
    if arr[n-1] > sum:
        return isSubsetSum (arr, n-1, sum)
 
    ''' else, check if sum can be obtained by any of 
    the following
    (a) including the last element
    (b) excluding the last element'''
     
    return isSubsetSum (arr, n-1, sum) or isSubsetSum (arr, n-1, sum-arr[n-1])
 
# following the implementation from https://www.geeksforgeeks.org/dynamic-programming-set-18-partition-problem/
def mypart(arr):
    if sum(arr)%2==0:
        return isSubsetSum(arr,len(arr),sum(arr)/2)
    else:
        return False

print(mypart([1,5,11,5]))
print(mypart([1,5,11,3]))
print(mypart([1,6,3,2]))





# -- end code --