import numpy as np
import matplotlib.pyplot as plt
temp = np.random.randn(9000) * 30 + 20
plt.hist(temp)
plt.show()

print(temp.mean())
print(temp.std())


temp = (temp-temp.mean())/temp.std()
plt.hist(temp)
plt.show()

print(temp.mean())
print(temp.std())