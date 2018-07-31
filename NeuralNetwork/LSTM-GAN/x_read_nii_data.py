import os,sys
import numpy as np
from nibabel.testing import data_path
from matplotlib import pyplot, cm
import nibabel as nib
import matplotlib.pyplot as plt

# read data
PathDicom = "../../Dataset/Neurofeedback_Skull_stripped/NFBS_Dataset/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".nii.gz" in filename.lower() and not 'brain' in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))


all_brain_data = np.zeros((10,192,256,256))

# for current_brain in range(len(lstFilesDCM)):
for current_brain in range(10):
    all_brain_data[current_brain] = nib.load(lstFilesDCM[current_brain]).get_fdata().T 

print(all_brain_data[0].max())
all_brain_data = all_brain_data/all_brain_data.max(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]
print(all_brain_data.max(axis=(1,2,3)))

sys.exit()

temp = lstFilesDCM[1]
img = nib.load(temp).get_fdata().T 
print(img.shape)
# img = img/img.max()
print(img.max())
print(img.min())

for x in img:
    plt.imshow(x,cmap='gray')
    plt.pause(0.01)

PathDicom = "../../Dataset/Neurofeedback_Skull_stripped/NFBS_Dataset/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".nii.gz" in filename.lower() and 'brainmask' in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

temp = lstFilesDCM[0]
img = nib.load(temp) 
print(img.shape)


# -- end code --