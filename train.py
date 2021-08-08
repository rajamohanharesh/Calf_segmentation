
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
import argparse




        



def get_arguments() -> argparse.Namespace:

	parser = argparse.ArgumentParser(
	    description="train a network for segmentation"
	)
	parser.add_argument("config", type=str, help="path to a config file")
	parser.add_argument(
	    "--seed",
	    type=int,
	    default=1234,
	    help="a number used to initialize a pseudorandom number generator.",
	)

	return parser.parse_args()

	def main() -> None:
	# argparser
	args = get_arguments()
	config = get_config(args.config)
	result_path = os.path.dirname(args.config)
	seed = args.seed

	warnings.filterwarnings('ignore')


	result_path = result_path + "/"+config.OBJECTIVE+"_segmentation_"+config.FOLD+"_FOLD_"+config.MODEL+"_model_"+"epochs_"+str(config.NUM_EPOCHS)+"_lr_"+str(config.LEARNING_RATE)+"_scaling_"+str(config.scaling)+"_rotation_"+str(config.rotation)+"_cropping_"+str(config.cropping)+"_flipping_"+str(config.flipping)+"_normalization_"+str(config.normalize)+"_dropout_"+str(config.DROPOUT_RATE)+"_beta_CE_"+config.beta_ce+"_beta_youden_"+config.beta_youden+"_beta_dice_"+config.beta_dice+"_BATCH_SIZE_"+str(config.BATCH_SIZE)+"_NUM_CHANNELS_"+str(config.NUM_CHANNELS)+"_NUM_CLASSES_"+str(config.NUM_CLASSES)+"/"
	if not os.path.exists(result_path):
	    os.makedirs(result_path)
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model_name = config.OBJECTIVE+"_segmentation_"+config.MODEL+"_model_"+"epochs_"+str(config.NUM_EPOCHS)+"_lr_"+str(config.LEARNING_RATE)+"_scaling_"+str(config.scaling)+"_rotation_"+str(config.rotation)+"_cropping_"+str(config.cropping)+"_flipping_"+str(config.flipping)+"_normalization_"+str(config.normalize)+"_dropout_"+str(config.DROPOUT_RATE)+"_beta_CE_"+config.beta_ce+"_beta_youden_"+config.beta_youden+"_beta_dice_"+config.beta_dice+"_BATCH_SIZE_"+str(config.BATCH_SIZE)+"_NUM_CHANNELS_"+str(config.NUM_CHANNELS)+"_NUM_CLASSES_"+str(config.NUM_CLASSES)

	if config.OBJECTIVE == "muscles":
		csv_path = "./muscles_csvs/"
	elif config.OBJECTIVE == "ROI":
		csv_path = "./ROI_csvs/"
	else:
		raise Exception("Onjective not implemented")

	OBJECTIVE = config.OBJECTIVE
    NUM_CLASSES = config.NUM_CLASSES
    NUM_CHANNELS = config.NUM_CHANNELS
    MODEL = config.MODEL
    BATCH_SIZE = config.BATCH_SIZE
    FOLD = config.FOLD
    LEARNING_RATE = config.LEARNING_RATE
    NUM_EPOCHS = config.NUM_EPOCHS
    DROPOUT_RATE = config.DROPOUT_RATE
    beta_dice = config.beta_dice
    beta_youden = config.beta_youden
    beta_ce = config.beta_ce
    scaling = config.scaling
    rotation = config.rotation
    cropping = config.cropping
    normalize = config.normalize
    flipping - config.flipping



	train_df = pd.read_csv(csv_path+"train"+str(FOLD)+".csv")
	val_df = pd.read_csv(csv_path+"val"+str(FOLD)+".csv")
	test_df = pd.read_csv(csv_path+"test"+str(FOLD)+".csv")

	

	train_loss_hist,val_loss_hist,train_dice_hist,val_dice_hist = train(train_df=val_df,val_df=val_df,NUM_CHANNELS=NUM_CHANNELS,NUM_CLASSES=NUM_CLASSES,normalize=normalize,flipping=flipping,rotation=rotation,BATCH_SIZE=BATCH_SIZE,NUM_EPOCHS=NUM_EPOCHS,modelPath = result_path)
	np.save(result_path+"train_loss.npy",train_loss_hist)
	np.save(result_path+"val_loss.npy",val_loss_hist)
	np.save(result_path+"train_dice.npy",train_dice_hist)
	np.save(result_path+"val_dice.npy",val_dice_hist)

	












