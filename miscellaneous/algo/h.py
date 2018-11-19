import numpy as np
import math
# array = np.array([3,2,5,1,1])
# print(minjump(0,len(array)))



def minjump(start,stop):
    count = 0
    if start == stop: return 0
    if start + 1 == stop: return 1

    # condition to exit fast
    if not result_mat[start,stop] == (stop) :return result_mat[start,stop]

    longg = array[start]
    minn = stop - start
    temp = 0
    i = 1

    while ( i <= longg and start+i <= stop ):
        print("BASIC OPERATION")
        temp = 1  + minjump(start+i,stop)
        if temp < minn: 
            result_mat[start,stop] = temp
            minn = temp
        i = i + 1

    return result_mat[start,stop]



print('-------------------')
array = np.array([3,5,2,1,1])
result_mat = np.ones((len(array),len(array))) * (len(array)-1) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([1,1,1,1,1])
result_mat = np.ones((len(array),len(array))) * (len(array)  -1) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')
print('================================')

print('-------------------')
array = np.array([1,0])
result_mat = np.ones((len(array),len(array))) * (len(array)-1 ) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([2,1,0])
result_mat = np.ones((len(array),len(array))) * (len(array) -1) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([3,2,1,0])
result_mat = np.ones((len(array),len(array))) * (len(array) -1) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([4,3,2,1,0])
result_mat = np.ones((len(array),len(array))) * (len(array) -1 ) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([5,4,3,2,1,0])
result_mat = np.ones((len(array),len(array))) * (len(array) -1 ) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')

print('-------------------')
array = np.array([6,5,4,3,2,1,0])
result_mat = np.ones((len(array),len(array))) * (len(array) -1 ) 
print('Length of array :',len(array))
print("FISRT BASIC OPERATION")
print('min jump :',minjump(0,len(array)-1))
print('-------------------')





















# -- end code --