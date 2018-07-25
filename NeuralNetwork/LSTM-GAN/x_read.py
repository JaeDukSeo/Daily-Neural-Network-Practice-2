import pydicom as dicom
import os
import numpy as np,sys
from matplotlib import pyplot as plt, cm
from collections import defaultdict
from scipy import ndimage
from PIL import Image, ImageSequence


# read the pet data as well as the CT dicom images
PathDicom = "../../Dataset/LSTM_GAN/z_gif.npz"
data = np.load(PathDicom)
print(type(data))
print(data.keys())

print(data.f.arr_0)
# print(dir(data))
# print(data['arr_r0'])
# print(data.shape)










# -- end code --