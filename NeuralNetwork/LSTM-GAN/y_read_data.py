import pydicom as dicom
import os
import numpy,sys
from matplotlib import pyplot, cm

# reading the dicom data
PathDicom = "../../Dataset/LSTM_GAN/RIDER_PHANTOM_PET_CT/"
lstFilesDCM = []  # create an empty list os.path.join(dirName,filename)

pet_scan_images = []
ct_scan_images = []
for dirName, subdirList, fileList in os.walk(PathDicom):
    if not len(fileList) == 0:
        if len(fileList) == 63:
            for filename in fileList:
                if ".dcm" in filename.lower():
                    ct_scan_images.append(os.path.join(dirName,filename))
        if len(fileList) == 47:
            for filename in fileList:
                if ".dcm" in filename.lower():
                    pet_scan_images.append(os.path.join(dirName,filename))


print(len(ct_scan_images))
print(len(pet_scan_images))

for ii in range(len(pet_scan_images)):
    print(ct_scan_images[ii])
    print(pet_scan_images[ii])


sys.exit()

for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".dcm" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

print(lstFilesDCM)
sys.exit()

# Get ref file
RefDs = dicom.read_file(lstFilesDCM[0])

# Load dimensions based on the number of rows, columns, and slices (along the Z axis)
ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

# Load spacing values (in mm)
ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), 
float(RefDs.SliceThickness))

# -- end code --