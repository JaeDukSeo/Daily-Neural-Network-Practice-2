import numpy as np

# coin row dynamic 
coin = [5,1,2,10,6,2]
maxval = [0] *  len(coin)
maxval[1] = coin[0]

for i in range(2, len(coin)):
    addcoin = coin[i] + maxval[i-2]
    if addcoin > maxval[i-1]: 
        maxval[i] = addcoin
    else:
        maxval[i] = maxval[i-1]
print(maxval)

# dynamic coin robot
matrix  = np.array([
    [0,0,0,0,1,0],
    [0,1,0,1,0,0],
    [0,0,0,1,0,1],
    [0,0,1,0,0,1],
    [1,0,0,0,1,0],
])

p = np.zeros_like(matrix)



# -- end code --