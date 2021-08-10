
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
from helper import train,evaluate,save_pred
from config import get_config
from losses import *





        



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


	result_path = result_path + "/"+config.OBJECTIVE+"_segmentation_"+str(config.FOLD)+"_FOLD_"+config.MODEL+"_model_"+"epochs_"+str(config.NUM_EPOCHS)+"_lr_"+str(config.LEARNING_RATE)+"_scaling_"+str(config.scaling)+"_rotation_"+str(config.rotation)+"_cropping_"+str(config.cropping)+"_flipping_"+str(config.flipping)+"_normalization_"+str(config.normalize)+"_dropout_"+str(config.DROPOUT_RATE)+"_beta_CE_"+str(config.beta_ce)+"_beta_youden_"+str(config.beta_youden)+"_beta_dice_"+str(config.beta_dice)+"_BATCH_SIZE_"+str(config.BATCH_SIZE)+"_NUM_CHANNELS_"+str(config.NUM_CHANNELS)+"_NUM_CLASSES_"+str(config.NUM_CLASSES)+"/"
	if not os.path.exists(result_path):
	    os.makedirs(result_path)
	

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
	flipping = config.flipping



	train_df = pd.read_csv(csv_path+"train"+str(FOLD)+".csv")
	val_df = pd.read_csv(csv_path+"val"+str(FOLD)+".csv")
	test_df = pd.read_csv(csv_path+"test"+str(FOLD)+".csv")

	loss_obj = Full_loss(beta_ce=beta_ce,beta_youden = beta_youden,beta_dice =beta_dice,NUM_CLASSES=NUM_CLASSES)




	save_pred(test_df,loss_obj.loss_fn,result_path,NUM_CHANNELS,NUM_CLASSES,normalize)

if __name__ == "__main__":
	main()

	












