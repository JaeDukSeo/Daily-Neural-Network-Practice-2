import numpy as np
import sys
np.random.seed(678)

# prob matrix of people moving from one side to another
prob = np.matrix([
    [0.7,0.2],
    [0.3,0.8]
])
print('---------')
print('The prob Matrix: ',prob)

print('---------')
print('The prob matrix after time')
print('After 2 Days : \n',prob ** 2)
print('After 30 Days : \n',prob ** 30)
print('After 365 Days : \n',prob ** 365)

print('---------')
print('---- Staring at Population 50 in Newyork, and 10 in California ----')
population = np.array([50,10])
print('After 2 Days : \n',(prob ** 2).dot(population))
print('After 30 Days : \n',(prob ** 30).dot(population))
print('After 365 Days : \n',(prob ** 365).dot(population))

sys.exit()

print('---------')
e_val,e_vec = np.linalg.eig(prob)
print('The eigen Value: ',e_val[1])
print('The eigen Vector: ',e_vec[:,1].T)

print('---------')
print('Started at [50,0]')
temp = np.array([50,10])
print((prob ** 10).dot(temp))
print((prob ** 100).dot(temp))
print((prob ** 1000).dot(temp))

e_veccc = e_vec[:,1]
e_veccc = np.array(e_veccc)
sumeed = e_veccc.sum()
e_veccc_div = e_veccc/e_veccc.sum()

print(e_veccc_div)
print(prob.dot(e_veccc_div))
print(temp.sum() * e_veccc_div)




# -- end code --