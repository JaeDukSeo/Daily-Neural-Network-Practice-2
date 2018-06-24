import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style="white")
np.random.seed(67)

# 0. Read the data
df = pd.read_csv('diabetes.csv')

# 1. Basic Description of the Data
print('------ Describe -------\n')
print(df.describe())
print('------ Info -------\n')
print(df.info())
print('------ Data Type -------\n')
print(df.dtypes)

# 1. See the first 10 and last 10 
print('------ First and Last 10 -------\n')
print(df.head())
print(df.tail())

# 2. clustermap heat map
# sns.heatmap(np.around(df.corr(),2),square=True,annot=True,linewidths=.2)
# plt.show()
# plt.close('all')

# 3. see the dist of the data
# df.hist(bins='auto',alpha=0.5)
# plt.show()
# plt.close('all')

# 4. 
import umap
from sklearn.manifold import TSNE


# -- end code --