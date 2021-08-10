import numpy as np
import pandas as pd
from scipy.ndimage import rotate
import tensorflow as tf








def rotate_data(X,y,angle):
    temp_rot_X = rotate(X,angle,reshape=False)
    temp_rot_X_2 = np.expand_dims(temp_rot_X, axis=0)

    temp_rot_y = rotate(y,angle,reshape=False)
    temp_rot_y_2 = np.expand_dims(temp_rot_y, axis=0)
    
    return temp_rot_X_2,temp_rot_y_2

def zeroMeanUnitVariance(input_image):
  # zero mean unit variance
    augmented_image = np.zeros(input_image.shape,dtype='float32')
    for ci in range(input_image.shape[0]):
        mn = np.mean(input_image[ci, ...])
        sd = np.std(input_image[ci, ...])
        augmented_image[ci, ...] = (input_image[ci, ...] - mn) / np.amax([sd, 1e-5])
    return augmented_image


def eroded(y,ROI):
    er = []
    for i in range(y.shape[0]):
        er.append(binary_erosion(y[i,...,ROI], iterations=2).astype(y.dtype))
    return np.asarray(er)

def fat_avg(x,y):
    fat_avgs = []
    fat_avgs.append(np.average(np.multiply(x[...,0],eroded(y,1))))
    fat_avgs.append(np.average(np.multiply(x[...,0],eroded(y,2))))
    fat_avgs.append(np.average(np.multiply(x[...,0],eroded(y,3))))
    fat_avgs.append(np.average(np.multiply(x[...,0],eroded(y,4))))
    fat_avgs.append(np.average(np.multiply(x[...,0],eroded(y,5))))
    return fat_avgs

def fat_fraction(patient_name, seg_path):
    fatmap_path = DATA_FOLDER + patient_name + '_fatmap.nii'
    if(os.path.exists(fatmap_path)):
        x, y, info = loadData_list_calf(fatmap_path,seg_path,1,0)
        return fat_avg(x,y)
    else:
        return -1