import numpy as np
import tensorflow as tf
import nibabel as nib
import keras
from utils import *



def load_data(train_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,train_mode=0):
    train_X, train_y, train_info = loadData_list_calf(train_df,NUM_CHANNELS,NUM_CLASSES)
    n_slices = train_X.shape[0]
    if normalize:
        train_X = zeroMeanUnitVariance(train_X)
    if flipping and train_mode:
        flip_X_1 = np.flip(train_X, 1)
        flip_y_1 = np.flip(train_y, 1)
        flip_X_2 = np.flip(train_X, 2)
        flip_y_2 = np.flip(train_y, 2)
        train_X = np.concatenate((train_X,flip_X_1,flip_X_2),axis=0)
        train_y = np.concatenate((train_y,flip_y_1,flip_y_2),axis=0)
    if rotation and train_mode:
        angles = np.random.uniform(low=0,high=360,size=6*n_slices)
        ii = 0
        for angle in angles:
            if ii == n_slices:
                ii=0
            temp_rot_X_2,temp_rot_y_2 = rotate_data(train_X[ii,...],train_y[ii,...],angle)

            train_X = np.concatenate((train_X,temp_rot_X_2),axis=0)
            train_y = np.concatenate((train_y,temp_rot_y_2),axis=0)
    return train_X,train_y, train_info




def loadData_list_calf(df,num_channels,num_classes):
    train_X = []
    train_y = []
    train_info = []

    for ii in range(len(df)):
        tmp_X = nib.load(str(df.iloc[ii]["Input"])).get_fdata()
        tmp_y = nib.load(str(df.iloc[ii]["GT"])).get_fdata()

        print(ii,tmp_X.shape,tmp_y.shape,str(df.iloc[ii]["Input"]),str(df.iloc[ii]["GT"]))
        for si in range(num_channels//2,tmp_X.shape[2]-num_channels//2,1):
            if np.sum(tmp_y[..., si])!=0:
                train_X.append(tmp_X[..., si-num_channels//2 : si + num_channels//2 +1])
                train_y.append(tmp_y[..., si])
                train_info.append('Filename:%s, slice:%d'%(df.iloc[ii]["Input"], si))
                print('Filename:%s, slice:%d'%(df.iloc[ii]["Input"], si))

    tmpp=np.asarray(train_X)
    return tmpp, tf.keras.utils.to_categorical(np.asarray(train_y), num_classes), train_info

def loadData_eval(df,num_channels,normalize,num_classes):
    train_list = []
    train_info = []

    for ii in range(len(df)):
        tmp_X = nib.load(str(df.iloc[ii]["Input"])).get_fdata()
        tmp_y = nib.load(str(df.iloc[ii]["GT"])).get_fdata()
        if normalize:
            tmp_X = zeroMeanUnitVariance(tmp_X)
        start = df.iloc[ii]["start"]
        end = df.iloc[ii]["end"]

        tmp_X = tmp_X[...,start:end]
        tmp_y = tmp_y[...,start:end]
        tmp_X = np.expand_dims(np.moveaxis(tmp_X, -1, 0),-1)
        tmp_y = np.moveaxis(tmp_y, -1, 0)
        #print(np.max(tmp_y),np.min(tmp_y),num_classes)

        train_list.append((tmp_X,tf.keras.utils.to_categorical(np.asarray(tmp_y).astype(int), int(num_classes))))
        train_info.append(df.iloc[ii]["ID"])
        
    
    return train_list, train_info

