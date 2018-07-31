import os
import numpy as np
from nibabel.testing import data_path
import os
import numpy
from matplotlib import pyplot, cm
import nibabel as nib

# read data
PathDicom = "../../Dataset/Neurofeedback_Skull_stripped/NFBS_Dataset/"
lstFilesDCM = []  # create an empty list
for dirName, subdirList, fileList in os.walk(PathDicom):
    for filename in fileList:
        if ".nii.gz" in filename.lower() and not 'brain' in filename.lower():  # check whether the file's DICOM
            lstFilesDCM.append(os.path.join(dirName,filename))

temp = lstFilesDCM[0]

# -- end code --