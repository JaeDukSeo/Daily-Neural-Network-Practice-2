import numpy as np
import sys



price = np.array([
     1   ,5  , 8 ,  9,  10,  17,  17,  20
])
dynamic = np.zeros_like(price)
dynamic[0] = 0
dynamic[1] = price[0]

def best2(n):
    if n==0: return 0
    max_valu = -9999
    for i in range(2,n):
        
        ress = price[i]

        for ll in range(i,n-i)

        temp = max(price[i] + dynamic[i-1],dynamic[i-2])
        if temp > max_valu: max_valu = temp
    return max_valu




size = len(price)
print(size)
print("Maximum Obtainable Value is", best2(size))



sys.exit()





# A Naive recursive solution 
# for Rod cutting problem
import sys
 
# A utility function to get the
# maximum of two integers
def max(a, b):
    return a if (a > b) else b
     
# Returns the best obtainable price for a rod of length n 
# and price[] as prices of different pieces
def cutRod(price, n):
    if(n <= 0):
        return 0
    max_val = -sys.maxsize-1
     
    # Recursively cut the rod in different pieces  
    # and compare different configurations
    for i in range(0, n):
        max_val = max(max_val, price[i] +
                      cutRod(price, n - i - 1))
    return max_val
 
# Driver code
arr = [1, 5, 8, 9, 10, 17, 17, 20]
size = len(arr)
print(size)
print("Maximum Obtainable Value is", cutRod(arr, size))
 
# This code is contributed by 'Smitha Dinesh Semwal'

price = np.array([
     1   ,5  , 8 ,  9,  10,  17,  17,  20
])
print(price)


def best1(n):
    if n <= 0 :return 0
    resul = price[n]
    for leng in range(0,n):
        ttry = price[leng] + best1(n-leng-1)
        if ttry > resul: resul = ttry
    return resul

# print(best1(0))
# print(best1(1))
print(best1(len(price)-1))




# -- end code ---