import numpy as np

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