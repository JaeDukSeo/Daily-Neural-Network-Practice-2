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
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()

sns.heatmap(corr)
plt.show()

# -- end code --