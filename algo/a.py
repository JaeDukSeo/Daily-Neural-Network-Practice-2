

ar = [2,3,4,4]
val=0
for i in range(len(ar)):
    powss = 0
    temp = 4
    for j in range(len(ar)-i): 
        powss = temp * temp
    val = val + ar[i] * powss
print(val)


# -- end code --