import numpy as np
import time,sys
n = 109.55
m = 10
d = np.array([1,2,5,20,40,100,200,400,1000,2000])

def makechange(amount,lastindex):
    best = [0] * m
    next_arr = [0] * m 

    if lastindex == 0:
        best[0] = amount / d[0]
        return best

    if not results_mat[lastindex-1,:].all()  == -1: return results_mat[lastindex-1,:] 

    best = makechange(amount,lastindex-1)
    i = 1
    while amount >= i * d[lastindex]:
        next_arr = makechange(amount - i*d[lastindex],lastindex-1)
        next_arr[lastindex] = i
        
        if numcoins(next_arr) <numcoins(best): 
            best = next_arr

            results_mat[lastindex-1,:]  = best

        i = i + 1 
    return results_mat[lastindex-1,:] 

def numcoins(x):return sum(x)



testing = np.array([1,2,4,2,5])
names = d * 5
howmanybills = 9
dollar = 2600//5

results_mat = np.ones(( howmanybills,howmanybills  )) * -1
print('Cent we want to exchange: ',dollar*5,' using how currency up to :', names[howmanybills])
print(makechange(dollar,howmanybills))

# for x in zip(names,makechange(dollar,howmanybills)):
#     print('we need : ',x[0],' cents number of cents we need : ',x[1])


# -- end code --