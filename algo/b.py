import numpy as np

def unique(x):
    for xx in x:
        if x.count(xx) > 1: return False
    return True
            

def closepoint(p):
    
    return_p1,return_p2 = None,None
    dmin = 999999999

    for i in range(len(p)):
        for j in range(i+1,len(p)):
            point1 = p[i] 
            point2 = p[j]

            d =  ( (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2 ) ** 0.5

            if d < dmin:
                dmin = d    
                return_p1 = point1 
                return_p2 = point2

    return return_p1,return_p2,dmin
            
    
# the answer is true
arr = [1,4,2,5,26,32]
print(unique(arr))

# the answer is [3,2] and [3,2]
ppp = [[1,2],[2,4],[8,3],[3,2],[3,2],[9,81]]
print(closepoint(ppp))

# the answer is false
arr = [1,4,2,5,26,32,1]
print(unique(arr))

arr = [1,4,2,5,26,32]
temp = []
for x in arr:
    temp.append([x,0])
cool = closepoint(temp)
print(cool)
if cool[0][0] == cool[1][1]: print("There is duplicate")
else: print('All element uni')


arr = [1,4,2,5,26,32,1]
temp = []
for x in arr:
    temp.append([x,0])
cool = closepoint(temp)
print(cool)
if cool[0][0] == cool[1][0]: print("There is duplicate")
else: print('All element uni')



arr = [1,4,2,5,26,32,2,24,2,53,2,1]
temp = []
for x in arr:
    temp.append([x,0])
cool = closepoint(temp)
print(cool)
if cool[0][0] == cool[1][0]: print("There is duplicate")
else: print('All element uni')



arr = [1,2,3,4,5,6,7,8,9,10]
temp = []
for x in arr:
    temp.append([x,0])
cool = closepoint(temp)
print(cool)
if cool[0][0] == cool[1][0]: print("There is duplicate")
else: print('All element uni')

# -- end code --
