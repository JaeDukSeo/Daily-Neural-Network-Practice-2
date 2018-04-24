import numpy as np
import math
# array = np.array([3,2,5,1,1])
# print(minjump(0,len(array)))


def minjump(start,stop):
    count = 0
    if start == stop: return 0
    if start + 1 == stop: return 1
    longg = array[start]
    minn = stop - start
    temp = 0
    i = 1
    while ( i <= longg and start+i <= stop ):
        print("BASIC OPERATION")
        temp = 1 + minjump(start+i,stop)
        if temp <minn: minn = temp
        i = i + 1
    return minn



print('-------------------')
array = np.array([3,5,2,1,1])
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([1,1,1,1,1])
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')
print('================================')

print('-------------------')
array = np.array([1,0])
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([2,1,0])
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([3,2,1,0])
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([4,3,2,1,0])
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([5,4,3,2,1,0])
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')























# -- garbage
import sys
sys.exit()
print('-------------------')
array = np.array([2, 3, 5, 8, 9, 2, 6, 7])
print('Length of array :',len(array))
print('min jump :',minjump(0,len(array)-1))
print('Factorail : ', math.factorial(len(array)))
print('-------------------')

print('-------------------')
array = np.array([3,2,5,1,1])
print('Length of array :',len(array))
print('min jump :',minjump(0,len(array)-1))
print('Factorail : ', math.factorial(len(array)))
print('-------------------')
print("=========================================")

print('-------BEST CASE length 4------------')
array = np.array([1,1,1,1])
print('Length of array :',len(array))
print('min jump :',minjump(0,len(array)-1))
print('Factorail : ', math.factorial(len(array)))
print('-------------------')

print('-------WORST CASE length 4------------')
array = np.array([5,4,3,2])
print('Length of array :',len(array))
print('min jump :',minjump(0,len(array)-1))
print('Factorail : ', math.factorial(len(array)))
print('-------------------')

print('---------length 4----------')
array = np.array([1,2,5,1])
print('Length of array :',len(array))
print('min jump :',minjump(0,len(array)-1))
print('Factorail : ', math.factorial(len(array)))
print('-------------------')


# -- end code --