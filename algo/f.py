import numpy as np


def exp1(x,n):
    if n == 0: return 1
    if n == 1: return x
    print('BASIC OPERATION!')
    return exp1(x,n//2) *exp1(x,n-n//2)

print(exp1(3,1)) # This should be 0
print('---------------------')
print(exp1(3,2)) # This should be 1
print('---------------------')
print(exp1(3,3)) # This should be 2
print('---------------------')
print(exp1(3,4)) # This should be 3
print('---------------------')
print(exp1(3,5)) # This should be 4
 

print('---------------------')
print('---------------------')


def exp2(x,n):
    if n == 0: return temp[n]
    if n == 1: return temp[n]
    if temp[n] == 0:
        temp[n]  = exp2(x,n//2) *exp2(x,n-n//2)
    return temp[n] 

n = 127
temp = [0] * (n+1)
temp[0],temp[1] = 0,3
print(exp2(3,n))    
print('---------------------')

n = 2
temp = [0] * (n+1)
temp[0],temp[1] = 0,3
print(exp2(3,n))    
print('---------------------')


n = 3
temp = [0] * (n+1)
temp[0],temp[1] = 0,3
print(exp2(3,n))    
print('---------------------')

n = 4
temp = [0] * (n+1)
temp[0],temp[1] = 0,3
print(exp2(3,n))    
print('---------------------')

n = 5
temp = [0] * (n+1)
temp[0],temp[1] = 0,3
print(exp2(3,n))    
print('---------------------')


# -- end code --