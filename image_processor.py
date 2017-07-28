import random
import numpy as np
from scipy import misc,ndimage

from image_processor_utils import *

def build_data_and_label(img, gt):
    '''
    For training.
    Returns random crop of image and ground truth.
    '''
    img = preprocess(img)

    img = scale_maxside(img, maxside=512)
    gt = scale_maxside(gt, maxside=512)
    
    # Random crop
    box = random_crop(img)
    img = crop_image(img, box)
    gt = crop_ground_truth(gt, box)

    # Batch size of 1
    data = img[np.newaxis, ...]
    label = gt[np.newaxis, ...]

    return data,label

def build_sliding_window(img):
    '''
    For testing.
    Returns sliding window patches as a batch.
    '''
    img = preprocess(img)
    img = scale_maxside(img, maxside=512)

    data = crop_sliding_window(img)
    return data

def post_process_sliding_window(img, crop_probs):
    '''
    For testing.
    Combines sliding window predictions into one image
    '''
    scaled = scale_maxside(img, maxside=512)
    probs = assemble_probs(scaled.shape, crop_probs)
    probs = scale(probs, img.shape)
    return probs
