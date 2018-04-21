
import math
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