import numpy as np
from medpy.metric.binary import assd
from medpy.metric.binary import hd



def dice(gt, pred):
    intersection = np.sum(np.multiply(gt, pred))
    union = np.sum(gt) + np.sum(pred) - intersection
    return (2 * intersection) / (union + intersection)

def voe(gt, pred):
    intersection = np.sum(np.multiply(gt, pred))
    union = np.sum(gt) + np.sum(pred) - intersection
    
    return (1-(intersection/union))


def compute_assd(gt,pred,voxelspacing,connectivity):

	return assd(pred,gt,voxelspacing=voxelspacing,connectivity=connectivity)


def compute_hd(gt,pred,voxelspacing,connectivity):
	
	return hd(pred,gt,voxelspacing=voxelspacing,connectivity=connectivity)