

def unique(x):
    for xx in x:
        if x.count(xx) > 1: return False
    return True
            


arr = [1,4,2,5,26,32]
print(unique(arr))





arr = [1,4,2,5,26,32,1]
print(unique(arr))




# -- end code --
