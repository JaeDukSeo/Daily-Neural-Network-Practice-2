import numpy as np

one = np.array([1,2,5,9])
two = np.array([0.2,0.44,0.02,0.1])
mean = 3

value = (one-mean)/two
print(value.sum())

value2 = (one/two) - (mean/two)
print(value2.sum())

value3 = (one/two).sum() - (mean/two).sum()
print(value3)

value4 = (one.sum()/two.sum()) - (mean/two.sum())
print(value4)
