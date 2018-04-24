import numpy as np

# array = np.array([3,2,5,1,1])

def minjump(start,stop):
    if start == stop: return 0
    if start + 1 == stop: return 1
    longg = array[start]
    minn = stop - start
    temp = 0 
    for i in range(1,longg):
        if start + i <= stop:
            temp = 1 + minjump(start+i,stop)
            if temp <minn: minn = temp
    return minn

# print(minjump(0,len(array)))


array = np.array([2, 3, 5, 8, 9, 2, 6, 7, 6, 8, 9])
print(minjump(0,len(array)))
# -- end code --