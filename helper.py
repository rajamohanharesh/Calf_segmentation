
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import nibabel as nib
from losses import *
from dataloader import *
from model import unet
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
import tensorflow as tf
from keras.models import load_model
from metrics import *
import os

def softmax(input,axis=-1):
  #print(np.exp(input))
  #print(np.sum(np.exp(input),axis=-1))
  return np.exp(input)/np.expand_dims(np.sum(np.exp(input),axis=-1),-1)


def train(train_df,val_df,full_cost,DROPOUT_RATE,LEARNING_RATE,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,BATCH_SIZE,NUM_EPOCHS,modelPath):
  train_X,train_y,train_info = load_data(train_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,train_mode=1)
  val_X,val_y,val_info = load_data(val_df,NUM_CHANNELS,NUM_CLASSES,normalize,flipping,rotation,train_mode=0)

  filepath = modelPath + "best_dice_model.hdf5"

  model = unet(input_size=(192,192,NUM_CHANNELS),NUM_CLASSES=NUM_CLASSES,DROPOUT_RATE=DROPOUT_RATE)
  mc = ModelCheckpoint(filepath, monitor='val_dice_multi', mode='min', verbose=1, save_best_only=True)

  model.compile(optimizer = Adam(lr = LEARNING_RATE), loss = full_cost, metrics = ['accuracy', youden, dice_multi, cross_entropy])
  model.summary()
  history = model.fit(train_X,
            train_y,
            validation_data = (val_X,val_y),
            batch_size = BATCH_SIZE,
            epochs = NUM_EPOCHS,
            callbacks=[mc],shuffle=True)

  return history.history["loss"],history.history["val_loss"],history.history["dice_multi"],history.history["val_dice_multi"]

def evaluate(val_df,full_cost,normalize,NUM_CHANNELS,NUM_CLASSES,result_path,connectivity=3,mode=0,fold=1,metrics=["dice","voe","assd","hd"]): #change the load function

  val_X,val_info = loadData_eval(val_df,num_channels=NUM_CHANNELS,num_classes=NUM_CLASSES,normalize=normalize)

  modelFile = result_path+"best_dice_model.hdf5"




  model = load_model(modelFile, custom_objects={'loss_fn':full_cost, 
                                                      'youden':youden, 
                                                      'dice_multi':dice_multi,
                                                      'cross_entropy':cross_entropy})

  gts = []
  preds = []
  for i in range(len(val_X)):
    #print(model.predict(val_X[i][0]))
    #print(model.predict(val_X[i][0]).shape)
    preds.append(softmax(model.predict(val_X[i][0]),axis=-1)) #more changes needed
    gts.append(val_X[i][1])
  #pred_categ, actual = flat_categ(predictions, val_y
  save_df(metrics,val_df, result_path,preds,gts,NUM_CLASSES,connectivity,mode,fold)



def save_df(metrics, val_df,result_path,predictions,ground_truths,num_classes,connectivity=3,mode=0,fold=1):
  for metric in metrics:
    if metric not in ["dice","voe","assd","hd"]:
      raise Exception("Metric not implemented")
    if metric == "hd":
      func = compute_hd
    elif metric =="assd":
      func = compute_assd
    elif metric =="dice":
      func = dice
    else:
      func = voe
    all_vals = []
    m=0
    for pred,gt in zip(predictions,ground_truths):
      voxelspacing = [val_df.iloc[m]["dim_1"],val_df.iloc[m]["dim_2"],val_df.iloc[m]["dim_3"]]
      m+=1


      values = []
      for n in range(num_classes):
        if metric in ["assd","hd"]:
          values.append(func(pred[...,n],gt[...,n],voxelspacing=voxelspacing,connectivity=connectivity))
        else:
          values.append(func(pred[...,n],gt[...,n]))
      all_vals.append(values)
      df =pd.DataFrame(data=np.array(all_vals),columns=[metric+"_"+str(i) for i in range(num_classes)])
      df["fold"] = fold
      df["train_mode"] = mode
      df["ID"] = val_df["ID"]

      df.to_csv(result_path+metric+".csv")








def save_pred(val_df,full_cost,result_path,NUM_CHANNELS,NUM_CLASSES,normalize):

  val_X,val_info = loadData_eval(val_df,num_channels=NUM_CHANNELS,num_classes=NUM_CLASSES,normalize=normalize)

  modelFile = result_path+"best_dice_model.hdf5"




  model = load_model(modelFile, custom_objects={'loss_fn':full_cost, 
                                                      'youden':youden, 
                                                      'dice_multi':dice_multi,
                                                      'cross_entropy':cross_entropy})
  save_path_preds = result_path+"predictions/"
  save_path_gts = result_path+"GTs/"

  if not os.path.exists(save_path_preds):
    os.makedirs(save_path_preds)
  if not os.path.exists(save_path_gts):
    os.makedirs(save_path_gts)


  for i in range(len(val_X)):
    voxelspacing = [val_df.iloc[i]["dim_1"],val_df.iloc[i]["dim_2"],val_df.iloc[i]["dim_3"]]

    output = softmax(model.predict(val_X[i][0]),axis=-1) #more changes needed
    GT = val_X[i][1]
    pred_new = np.argmax(output,-1)
    name = val_df.iloc[i]["ID"]
    img_pred=nib.Nifti1Image(pred_new,affine=np.eye(4))
    img_GT=nib.Nifti1Image(pred_new,affine=np.eye(4))

    img_pred.header['pixdim'][1:4]  = voxelspacing
    img_GT.header['pixdim'][1:4]  = voxelspacing

    nib.save(img_pred,save_path_preds+name+"_pred.nii.gz")
    nib.save(img_GT,save_path_gts+name+"_GT.nii.gz")








 #change the load function


			

