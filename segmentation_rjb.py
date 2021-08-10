
import numpy as np
import sys
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from pathlib import Path
import time
import nibabel as nib
from datetime import datetime
import os
import skimage.io as io
import skimage.transform as trans
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from scipy.ndimage import rotate
from sklearn.metrics import confusion_matrix
import pandas as pd
from tensorflow.keras.utils import to_categorical
from scipy import ndimage
from random import uniform
from scipy.ndimage import distance_transform_edt as distance
from scipy.ndimage.morphology import binary_erosion


# In[12]:


# initialize constants
NUM_CLASSES = 6
NUM_CHANNELS = 9
NUM_EPOCHS = 400
BATCH_SIZE = 8
DROPOUT_RATE = 0
LEARNING_RATE = 1e-4
seed = 1234
beta_dice = 0
beta_youden = 1
beta_ce = 0.1


'''
Options for cost:
- 'multi_dice_cost'
- 'youden and cross entropy'
'''

loss_fn = beta_dice*dice_multi + beta_youden*youden + beta_ce*cross_entropy



DATA_FOLDER = '/gpfs/data/denizlab/Users/jts602/v3_rjb_fat_192_192_144'
#DATA_FOLDER = '/gpfs/data/denizlab/Datasets/Calf/v2_rjb_192_192_144'
dtype = tf.float32
cv = [0,1,2]

dateTimeObj = datetime.now()
timestampStr = dateTimeObj.strftime("%Y-%m-%d-%H:%M")
path = os.getcwd()
folderPath = path + '/models/' + timestampStr + '_rjb'
print('Folder Path: ', folderPath)

if not os.path.exists(folderPath):
    os.mkdir(folderPath)

printOutputPath = folderPath + '/print_output.txt'
sys.stdout = open(printOutputPath, "w")
print('NUM_CLASSES: ' + str(NUM_CLASSES) + '\n')
print('NUM_CHANNELS: ' + str(NUM_CHANNELS) + '\n')
print('NUM_EPOCHS: ' + str(NUM_EPOCHS) + '\n')
print('BATCH_SIZE: ' + str(BATCH_SIZE) + '\n')
print('DROPOUT_RATE: ' + str(DROPOUT_RATE) + '\n')
print('ROTATION ANGLES: 0 to 360 separated by 60 degrees' + '\n')
print('COST: ' + cost + '\n')


# ## Cost Function Definition

# In[13]:





# ## Defining the Model
# 
# Here, we employ a u-net architecture for this multi-class segmentation task.

# In[17]:



# ## Loading Data and Training the Model
# 
# Here, we load the data, normalize it, and split it into training and test sets. Then, we train the data.
# 
# <b>Data augmentations done:</b>
# - vertical flip
# - horizontal flip
# - random rotations
# 
# <b>Data augmentations to add:</b>
# - shift (0.2 of image height and width)
# - zoom between 0.9 to 1.3 of image size

# In[20]:







# In[21]:





# In[22]:



    
    
    


# In[23]:


# organize the data into training and validation sets

skf = StratifiedKFold(shuffle=True, n_splits=3, random_state=seed)
for i,(train_index, test_index) in enumerate(skf.split(np.zeros(len(X)), np.zeros(len(X)))):
    if i in cv:
        print("TRAIN:", train_index, "VAL:", test_index)
        X_train = list( X[j] for j in train_index )
        y_train = list( y[j] for j in train_index )
        n_samples = len(X_train)
        X_val = list( X[j] for j in test_index )
        y_val = list( y[j] for j in test_index )
        
        # obtain slices
        train_X, train_y, train_info = loadData_list_calf(X_train,y_train,NUM_CHANNELS,0)
        n_slices = train_X.shape[0]
        train_X = zeroMeanUnitVariance(train_X)
        
        # flip data augmentation
        flip_X_1 = np.flip(train_X, 1)
        print(flip_X_1.shape)
        flip_y_1 = np.flip(train_y, 1)
        flip_X_2 = np.flip(train_X, 2)
        flip_y_2 = np.flip(train_y, 2)
        train_X = np.concatenate((train_X,flip_X_1,flip_X_2),axis=0)
        train_y = np.concatenate((train_y,flip_y_1,flip_y_2),axis=0)
        
        # rotate data augmentation
        for ii in range(n_slices):
            
            angle = uniform(0,60)
            
            
            temp_rot_X_2,temp_rot_y_2 = rotate_data(train_X[ii,...],train_y[ii,...],angle)

            train_X = np.concatenate((train_X,temp_rot_X_2),axis=0)
            train_y = np.concatenate((train_y,temp_rot_y_2),axis=0)
            
        for iii in range(n_slices):
            
            angle = uniform(60,120)
            
            temp_rot_X_2,temp_rot_y_2 = rotate_data(train_X[iii,...],train_y[iii,...],angle)

            train_X = np.concatenate((train_X,temp_rot_X_2),axis=0)
            train_y = np.concatenate((train_y,temp_rot_y_2),axis=0)
            
        for iv in range(n_slices):
            
            angle = uniform(120,180)
            
            temp_rot_X_2,temp_rot_y_2 = rotate_data(train_X[iv,...],train_y[iv,...],angle)

            train_X = np.concatenate((train_X,temp_rot_X_2),axis=0)
            train_y = np.concatenate((train_y,temp_rot_y_2),axis=0)
            
        for v in range(n_slices):
            
            angle = uniform(180,240)
            
            temp_rot_X_2,temp_rot_y_2 = rotate_data(train_X[v,...],train_y[v,...],angle)

            train_X = np.concatenate((train_X,temp_rot_X_2),axis=0)
            train_y = np.concatenate((train_y,temp_rot_y_2),axis=0)
            
        for vi in range(n_slices):
            
            angle = uniform(240,300)
            
            temp_rot_X_2,temp_rot_y_2 = rotate_data(train_X[vi,...],train_y[vi,...],angle)

            train_X = np.concatenate((train_X,temp_rot_X_2),axis=0)
            train_y = np.concatenate((train_y,temp_rot_y_2),axis=0)
            
        for vii in range(n_slices):
            
            angle = uniform(300,360)
            
            temp_rot_X_2,temp_rot_y_2 = rotate_data(train_X[vii,...],train_y[vii,...],angle)

            train_X = np.concatenate((train_X,temp_rot_X_2),axis=0)
            train_y = np.concatenate((train_y,temp_rot_y_2),axis=0)
        
        # shift data augmentation
        
        print('Shape of training data:',train_X.shape,train_y.shape)
        del X_train, y_train
    
        val_X, val_y, val_info = loadData_list_calf(X_val, y_val,NUM_CHANNELS,0)
        val_X = zeroMeanUnitVariance(val_X)
        print('Shape of validation data:',val_X.shape,val_y.shape)
        del X_val, y_val
                
        # train the model
        os.mkdir(folderPath + '/CV' + str(i))
        modelPath = folderPath + '/CV' + str(i) + '/model'
        model = unet()
        mc = ModelCheckpoint(modelPath, monitor='val_dice_multi', mode='min', verbose=1, save_best_only=True)

        model.compile(optimizer = Adam(lr = LEARNING_RATE), loss = full_cost, metrics = ['accuracy', youden, dice_multi, cross_entropy])
        model.summary()
        model.fit(train_X,
                  train_y,
                  validation_data = (val_X,val_y),
                  batch_size = BATCH_SIZE,
                  epochs = NUM_EPOCHS,
                  callbacks=[mc])

        # model.save(folderPath + '/CV' + str(i) + '/model')


# ## Dice Similarity Coefficient Analysis

# In[1]:


from medpy.metric.binary import assd


# In[4]:


from medpy.metric.binary import assd
from medpy.metric.binary import hd




    


# In[5]:


def all_dice(gt_categ, pred_categ, df, cv, val):
    df = df.append({'cv': cv,
                    'test': val,
                    'dsc:0 (background)': dice(gt_categ[...,0], pred_categ[...,0]), 
                    'dsc:1 (gastrocnemius medial)': dice(gt_categ[...,1], pred_categ[...,1]), 
                    'dsc:2 (gastrocnemius lateral)': dice(gt_categ[...,2], pred_categ[...,2]), 
                    'dsc:3 (soleus)': dice(gt_categ[...,3], pred_categ[...,3]), 
                    'dsc:4 (tibia)': dice(gt_categ[...,4], pred_categ[...,4]), 
                    'dsc:5 (fibula)': dice(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)
    return df

def slices_dice(gt_categ, pred_categ, df, cv, val, info):
    df = df.append({'cv': cv,
                    'test': val,
                    'info': info,
                    'dsc:0 (background)': dice(gt_categ[...,0], pred_categ[...,0]), 
                    'dsc:1 (gastrocnemius medial)': dice(gt_categ[...,1], pred_categ[...,1]), 
                    'dsc:2 (gastrocnemius lateral)': dice(gt_categ[...,2], pred_categ[...,2]), 
                    'dsc:3 (soleus)': dice(gt_categ[...,3], pred_categ[...,3]), 
                    'dsc:4 (tibia)': dice(gt_categ[...,4], pred_categ[...,4]), 
                    'dsc:5 (fibula)': dice(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)
    return df
def all_voe(gt_categ, pred_categ, df, cv, val):
    df = df.append({'cv': cv,
                    'test': val,
                    'voe:0 (background)': voe(gt_categ[...,0], pred_categ[...,0]), 
                    'voe:1 (gastrocnemius medial)': voe(gt_categ[...,1], pred_categ[...,1]), 
                    'voe:2 (gastrocnemius lateral)': voe(gt_categ[...,2], pred_categ[...,2]), 
                    'voe:3 (soleus)': voe(gt_categ[...,3], pred_categ[...,3]), 
                    'voe:4 (tibia)': voe(gt_categ[...,4], pred_categ[...,4]), 
                    'voe:5 (fibula)': voe(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)
    return df

def slices_voe(gt_categ, pred_categ, df, cv, val, info):
    df = df.append({'cv': cv,
                    'test': val,
                    'info': info,
                    'voe:0 (background)': voe(gt_categ[...,0], pred_categ[...,0]), 
                    'voe:1 (gastrocnemius medial)': voe(gt_categ[...,1], pred_categ[...,1]), 
                    'voe:2 (gastrocnemius lateral)': voe(gt_categ[...,2], pred_categ[...,2]), 
                    'voe:3 (soleus)': voe(gt_categ[...,3], pred_categ[...,3]), 
                    'voe:4 (tibia)': voe(gt_categ[...,4], pred_categ[...,4]), 
                    'voe:5 (fibula)': voe(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)
    return df

def all_assd(gt_categ, pred_categ, df, cv, val):
    df = df.append({'cv': cv,
                    'test': val,
                    'assd:0 (background)': compute_assd(gt_categ[...,0], pred_categ[...,0]), 
                    'assd:1 (gastrocnemius medial)': compute_assd(gt_categ[...,1], pred_categ[...,1]), 
                    'assd:2 (gastrocnemius lateral)': compute_assd(gt_categ[...,2], pred_categ[...,2]), 
                    'assd:3 (soleus)': compute_assd(gt_categ[...,3], pred_categ[...,3]), 
                    'assd:4 (tibia)': compute_assd(gt_categ[...,4], pred_categ[...,4]), 
                    'assd:5 (fibula)': compute_assd(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)
    return df

def slices_assd(gt_categ, pred_categ, df, cv, val, info):
    df = df.append({'cv': cv,
                    'test': val,
                    'info': info,
                    'assd:0 (background)': compute_assd(gt_categ[...,0], pred_categ[...,0]), 
                    'assd:1 (gastrocnemius medial)': compute_assd(gt_categ[...,1], pred_categ[...,1]), 
                    'assd:2 (gastrocnemius lateral)': compute_assd(gt_categ[...,2], pred_categ[...,2]), 
                    'assd:3 (soleus)': compute_assd(gt_categ[...,3], pred_categ[...,3]), 
                    'assd:4 (tibia)': compute_assd(gt_categ[...,4], pred_categ[...,4]), 
                    'assd:5 (fibula)': compute_assd(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)
def all_hd(gt_categ, pred_categ, df, cv, val):
    df = df.append({'cv': cv,
                    'test': val,
                    'hd:0 (background)': compute_hd(gt_categ[...,0], pred_categ[...,0]), 
                    'hd:1 (gastrocnemius medial)': compute_hd(gt_categ[...,1], pred_categ[...,1]), 
                    'hd:2 (gastrocnemius lateral)': compute_hd(gt_categ[...,2], pred_categ[...,2]), 
                    'hd:3 (soleus)': compute_hd(gt_categ[...,3], pred_categ[...,3]), 
                    'hd:4 (tibia)': compute_hd(gt_categ[...,4], pred_categ[...,4]), 
                    'hd:5 (fibula)': compute_hd(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)
    return df

def slices_hd(gt_categ, pred_categ, df, cv, val, info):
    df = df.append({'cv': cv,
                    'test': val,
                    'info': info,
                    'hd:0 (background)': compute_hd(gt_categ[...,0], pred_categ[...,0]), 
                    'hd:1 (gastrocnemius medial)': compute_hd(gt_categ[...,1], pred_categ[...,1]), 
                    'hd:2 (gastrocnemius lateral)': compute_hd(gt_categ[...,2], pred_categ[...,2]), 
                    'hd:3 (soleus)': compute_hd(gt_categ[...,3], pred_categ[...,3]), 
                    'hd:4 (tibia)': compute_hd(gt_categ[...,4], pred_categ[...,4]), 
                    'hd:5 (fibula)': compute_hd(gt_categ[...,5], pred_categ[...,5])},
                    ignore_index=True)



# In[19]:





# In[20]:


# pandas dataframe to store dice scores
df_dice = pd.DataFrame(columns=['cv', 
                           'test', 
                           'dsc:0 (background)', 
                           'dsc:1 (gastrocnemius medial)', 
                           'dsc:2 (gastrocnemius lateral)', 
                           'dsc:3 (soleus)', 
                           'dsc:4 (tibia)', 
                           'dsc:5 (fibula)'])

df_slices_dice = pd.DataFrame(columns=['cv', 
                           'test',
                           'info',
                           'dsc:0 (background)', 
                           'dsc:1 (gastrocnemius medial)', 
                           'dsc:2 (gastrocnemius lateral)', 
                           'dsc:3 (soleus)', 
                           'dsc:4 (tibia)', 
                           'dsc:5 (fibula)'])

df_voe = pd.DataFrame(columns=['cv', 
                           'test', 
                           'voe:0 (background)', 
                           'voe:1 (gastrocnemius medial)', 
                           'voe:2 (gastrocnemius lateral)', 
                           'voe:3 (soleus)', 
                           'voe:4 (tibia)', 
                           'voe:5 (fibula)'])

df_slices_voe = pd.DataFrame(columns=['cv', 
                           'test',
                           'info',
                           'voe:0 (background)', 
                           'voe:1 (gastrocnemius medial)', 
                           'voe:2 (gastrocnemius lateral)', 
                           'voe:3 (soleus)', 
                           'voe:4 (tibia)', 
                           'voe:5 (fibula)'])
df_assd = pd.DataFrame(columns=['cv', 
                           'test', 
                           'assd:0 (background)', 
                           'assd:1 (gastrocnemius medial)', 
                           'assd:2 (gastrocnemius lateral)', 
                           'assd:3 (soleus)', 
                           'assd:4 (tibia)', 
                           'assd:5 (fibula)'])

df_slices_assd = pd.DataFrame(columns=['cv', 
                           'test',
                           'info',
                           'assd:0 (background)', 
                           'assd:1 (gastrocnemius medial)', 
                           'assd:2 (gastrocnemius lateral)', 
                           'assd:3 (soleus)', 
                           'assd:4 (tibia)', 
                           'assd:5 (fibula)'])

df_hd = pd.DataFrame(columns=['cv', 
                           'test', 
                           'hd:0 (background)', 
                           'hd:1 (gastrocnemius medial)', 
                           'hd:2 (gastrocnemius lateral)', 
                           'hd:3 (soleus)', 
                           'hd:4 (tibia)', 
                           'hd:5 (fibula)'])

df_slices_hd = pd.DataFrame(columns=['cv', 
                           'test',
                           'info',
                           'hd:0 (background)', 
                           'hd:1 (gastrocnemius medial)', 
                           'hd:2 (gastrocnemius lateral)', 
                           'hd:3 (soleus)', 
                           'hd:4 (tibia)', 
                           'hd:5 (fibula)'])



fold = 1

train_df = pd.read_csv(csv_path+"train"+str(fold)+".csv")
val_df = pd.read_csv(csv_path+"val"+str(fold)+".csv")
test_df = pd.read_csv(csv_path+"test"+str(fold)+".csv")



# get dice scores for each cv
skf = StratifiedKFold(shuffle=True, n_splits=3, random_state=seed)
for i,(train_index, test_index) in enumerate(skf.split(np.zeros(len(X)), np.zeros(len(X)))):
    if i in cv:
        modelPath = folderPath + '/CV' + str(i) + '/model'
        model = load_model(modelPath, custom_objects={'full_cost':full_cost, 
                                                      'youden':youden, 
                                                      'dice_multi':dice_multi,
                                                      'cross_entropy':cross_entropy})
        
        print("TRAIN:", train_index, "VAL:", test_index)
        X_train = list( X[i] for i in train_index )
        y_train = list( y[i] for i in train_index )
        n_samples = len(X_train)
        X_val = list( X[i] for i in test_index )
        y_val = list( y[i] for i in test_index )
        
        # obtain slices
        train_X, train_y, train_info = loadData_list_calf(X_train,y_train,NUM_CHANNELS,0)
        n_slices = train_X.shape[0]
        train_X = zeroMeanUnitVariance(train_X)
        
        print('Shape of training data:',train_X.shape,train_y.shape)
        del X_train, y_train
    
        val_X, val_y, val_info = loadData_list_calf(X_val, y_val,NUM_CHANNELS,0)
        val_n_slices = val_X.shape[0]
        val_X = zeroMeanUnitVariance(val_X)
        print('Shape of validation data:',val_X.shape,val_y.shape)
        del X_val, y_val
        
        predictions = model.predict(val_X)
        pred_categ, actual = flat_categ(predictions, val_y)
        df_dice = all_dice(actual, pred_categ, df_dice, i, 1)
        df_voe = all_dice(actual, pred_categ,df_voe , i, 1)
        df_assd = all_dice(actual, pred_categ, df_assd, i, 1)
        df_hd = all_dice(actual, pred_categ, df_hd, i, 1)
        for ii in range(0,val_n_slices):
            val_pred_categ_slice, val_actual_slice = flat_categ(predictions[ii,...],val_y[ii,...])
            df_slices_dice = slices_dice(val_actual_slice, val_pred_categ_slice, df_slices_dice, i, 1, val_info[ii])
            df_slices_voe = slices_voe(val_actual_slice, val_pred_categ_slice, df_slices_voe, i, 1, val_info[ii])
            df_slices_assd = slices_assd(val_actual_slice, val_pred_categ_slice, df_slices_assd, i, 1, val_info[ii])
            df_slices_hd = slices_hd(val_actual_slice, val_pred_categ_slice, df_slices_hd, i, 1, val_info[ii])
                
        train_predictions = model.predict(train_X)
        train_pred_categ, train_actual = flat_categ(train_predictions,train_y)
        df_dice = all_dice(train_actual, pred_categ, df_dice, i, 0)
        df_voe = all_dice(train_actual, pred_categ,df_voe , i, 0)
        df_assd = all_dice(train_actual, pred_categ, df_assd, i, 0)
        df_hd = all_dice(train_actual, pred_categ, df_hd, i, 0)
        for iii in range(0,n_slices):
            train_pred_categ_slice, train_actual_slice = flat_categ(train_predictions[iii,...],train_y[iii,...])
            df_slices_dice = slices_dice(train_actual_slice, train_pred_categ_slice, df_slices_dice, i, 0, train_info[iii])
            df_slices_voe = slices_voe(train_actual_slice, train_pred_categ_slice, df_slices_voe, i, 0, train_info[iii])
            df_slices_assd = slices_assd(train_actual_slice, train_pred_categ_slice, df_slices_assd, i, 0, train_info[iii])
            df_slices_hd = slices_hd(train_actual_slice, train_pred_categ_slice, df_slices_hd, i, 0, train_info[iii])

        #print(df_dice)
        
df_dice.to_csv(folderPath + '/dsc_' + timestampStr + '.csv')
df_slices_dice.to_csv(folderPath + '/dsc_slices' + timestampStr + '.csv')
df_voe.to_csv(folderPath + '/voe_' + timestampStr + '.csv')
df_slices_voe.to_csv(folderPath + '/voe_slices' + timestampStr + '.csv')
df_assd.to_csv(folderPath + '/assd_' + timestampStr + '.csv')
df_slices_assd.to_csv(folderPath + '/assd_slices' + timestampStr + '.csv')
df_hd.to_csv(folderPath + '/hd_' + timestampStr + '.csv')
df_slices_hd.to_csv(folderPath + '/hd_slices' + timestampStr + '.csv')


# In[ ]:


assd()


# ## Inference and Fat Fraction Analysis

# In[21]:


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


# In[22]:


def inference(model, X_val, y_val, cv, test):
    for ii in range(len(X_val)):
        
        tmp_X = nib.load(str(X_val[ii]))
        
        filestr = str(X_val[ii])
        start = filestr.find('144/')
        end = filestr.find('_mri', start)
        patient_name = filestr[start+4:end]
        print(patient_name)
        
        val_X, val_y, val_info = loadData_list_calf(X_val[ii],y_val[ii],NUM_CHANNELS,1)
        val_X = zeroMeanUnitVariance(val_X)
        prediction = model.predict(val_X)
        print('prediction shape:', prediction.shape) # 136 x 192 x 192 x 6
        
        tf_flat_pred = tf.reshape(prediction, [-1, NUM_CLASSES])
        tf_flat_pred_2 = tf.nn.softmax(tf_flat_pred)
        flat_pred = tf_flat_pred_2.eval(session=tf.compat.v1.Session())    
        pred = np.argmax(flat_pred, axis=1)
        reshaped = np.reshape(pred, (-1,192,192))
        channels_arr = np.zeros((9//2, 192, 192))
        tmp_matrix = np.concatenate((channels_arr, reshaped, channels_arr),axis=0)
        outMatrix = np.swapaxes((np.swapaxes(tmp_matrix,0,1)),1,2)        
        img = nib.Nifti1Image(outMatrix, affine = tmp_X.affine)
        if test == 1:
            save_to_path = folderPath + '/CV' + cv + '/test/inference_' + str(ii) + '_' + patient_name + '_' + timestampStr + '.nii'
            print(save_to_path)
            nib.save(img, save_to_path)
            fat_fraction(patient_name, save_to_path)
        else:
            save_to_path = folderPath + '/CV' + cv + '/train/inference_' + str(ii) + '_' + patient_name + '_' + timestampStr + '.nii'
            print(save_to_path)
            nib.save(img, save_to_path)


# In[23]:


skf = StratifiedKFold(shuffle=True, n_splits=3, random_state=seed)
for i,(train_index, test_index) in enumerate(skf.split(np.zeros(len(X)), np.zeros(len(X)))):
    if i in cv:
        modelPath = folderPath + '/CV' + str(i) + '/model'
        model = load_model(modelPath, custom_objects={'full_cost':full_cost, 
                                                      'youden':youden, 
                                                      'dice_multi':dice_multi,
                                                      'cross_entropy':cross_entropy})
        
        print("TRAIN:", train_index, "VAL:", test_index)
        X_train = list( X[j] for j in train_index )
        y_train = list( y[j] for j in train_index )
        X_val = list( X[j] for j in test_index )
        y_val = list( y[j] for j in test_index )
            
        os.mkdir(folderPath + '/CV' + str(i) + '/train')
        os.mkdir(folderPath + '/CV' + str(i) + '/test')
        
        inference(model, X_val, y_val, str(i), 1)
        inference(model, X_train, y_train, str(i), 0)



sys.stdout.close()

