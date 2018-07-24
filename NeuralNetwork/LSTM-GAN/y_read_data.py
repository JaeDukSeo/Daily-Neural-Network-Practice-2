import pydicom as dicom
import os
import numpy,sys
from matplotlib import pyplot, cm
from collections import defaultdict





# read the pet data as well as the CT dicom images
PathDicom = "../../Dataset/LSTM_GAN/RIDER_PHANTOM_PET_CT/"
lstFilesDCM = []  # create an empty list os.path.join(dirName,filename)
pet_scan_images = defaultdict(list);pet_scan_images_count = 0
ct_scan_images = defaultdict(list);ct_scan_images_count = 0
for dirName, subdirList, fileList in os.walk(PathDicom):
    if not len(fileList) == 0:
        
        if len(fileList) == 63:
            temp = []
            for filename in fileList:
                if ".dcm" in filename.lower():
                    temp.append(os.path.join(dirName,filename))
            ct_scan_images[ct_scan_images_count].append(temp)
            ct_scan_images_count = ct_scan_images_count + 1

        if len(fileList) == 47:
            temp = []
            for filename in fileList:
                if ".dcm" in filename.lower():
                    temp.append(os.path.join(dirName,filename))
                    
            pet_scan_images[pet_scan_images_count].append(temp)
            pet_scan_images_count = pet_scan_images_count + 1



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