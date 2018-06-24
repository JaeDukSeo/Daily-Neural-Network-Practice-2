import pandas as pd
import matplotlib.pyplot as plt

# 0. Read the data
df = pd.read_csv('diabetes.csv')

# 1. Basic Description of the Data
print(df.describe())
print(df.info())
print(df.dtypes)

# 2. clustermap heat map
import seaborn as sns
sns.set(style="white")
sns.heatmap(df.corr())
plt.show()

# -- end code --