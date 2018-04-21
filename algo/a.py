
import math











def fib(x):
    if x == 0 : return 0
    if x ==1 : return 1
    return fib(x-1) + fib(x-2)

# print(fib(100))


temp = [-1] * 101
temp[0]  = 0
temp[1] = 1

def fibb(x):
    if temp[x] == -1: 
        temp[x] =  fibb(x-1) + fibb(x-2) 
    return temp[x]

print(fibb(100))







import sys
sys.exit()
# --------
ar = [2,3,4,4]
val=0
for i in range(len(ar)):
    powss = 0
    temp = 4
    for j in range(len(ar)-i): 
        powss = temp * temp
    val = val + ar[i] * powss
print(val)


def binar(n):
    count = 1
    while n > 1: 
        count = count + 1
        n = math.floor(n/2)
    return count

print(binar(4))
print(binar(2))
print(binar(3))
print(binar(4))
print(binar(5))



# -- end code --