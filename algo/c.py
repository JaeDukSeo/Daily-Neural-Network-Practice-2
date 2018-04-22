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

f = np.zeros_like(matrix)
n,m = matrix.shape

print(matrix)
print('--------------')
for j in range(m):
    f[1,j] = f[1,j] + matrix[1,j]
print(f)
print('--------------')
for i in range(n):
    f[i,1] = f[i-1,1] + matrix[i,1]

    for j in range(m):
        
        if f[i-1,j] > f[i,j-1]:
            f[i,j] = f[i-1,j] + matrix[i,j]
        else:
            f[i,j] = f[i,j-1] + matrix[i,j]
print('--------------')
print(f)

# -- end code --