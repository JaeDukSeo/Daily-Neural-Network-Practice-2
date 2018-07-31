import pydicom as dicom
import os
import numpy as np,sys
from matplotlib import pyplot as plt, cm
from collections import defaultdict
from scipy import ndimage
from PIL import Image, ImageSequence


# read the pet data as well as the CT dicom images
PathDicom = "../../Dataset/LSTM_GAN/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".gif" in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

all_data = []
for x in range(len(lstFilesDCM)):
    img = Image.open(lstFilesDCM[x])
    frames = np.array([np.array(frame.copy().convert('RGB').getdata(),dtype=np.uint8).reshape(frame.size[1],frame.size[0],3) for frame in ImageSequence.Iterator(img)])
    all_data.append(frames)
    print(frames.shape) 
    print(len(all_data))

# for xx in all_data:
#     for xxx in xx:
#         plt.imshow(xxx)
#         plt.pause(0.3)


all_data = np.asarray(all_data)
print(all_data.shape) 
np.save('gif.npy', all_data)

sys.exit()

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
            ct_scan_images[ct_scan_images_count] = temp
            ct_scan_images_count = ct_scan_images_count + 1

        if len(fileList) == 47:
            temp = []
            for filename in fileList:
                if ".dcm" in filename.lower():
                    temp.append(os.path.join(dirName,filename))
                    
            pet_scan_images[pet_scan_images_count] = temp
            pet_scan_images_count = pet_scan_images_count + 1


print(len(pet_scan_images))
print(len(pet_scan_images[0]))

for xx in pet_scan_images:
    current_pet = pet_scan_images[xx]
    for xxx in range(len(current_pet)):
        curren_pet_image = dicom.read_file(current_pet[xxx]).pixel_array
        plt.title(str(current_pet[xxx][-10:]))
        plt.imshow(curren_pet_image,cmap='gray')
        plt.pause(0.1)

# read the dicom to numpy
# ct_scan_numpys = np.zeros(20,)

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