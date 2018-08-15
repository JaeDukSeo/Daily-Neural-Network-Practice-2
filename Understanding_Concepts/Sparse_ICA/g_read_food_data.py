import pandas as pd

temp = pd.read_hdf('../../Dataset/food_data/food_test_c101_n1000_r128x128x1.h5','images')
print(temp.shape)
