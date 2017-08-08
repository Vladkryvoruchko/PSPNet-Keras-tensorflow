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

    #if np.ndim(gt) == 2:
    #    gt = gt[..., np.newaxis]

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


if __name__ == "__main__":
    img = misc.imread("../ADE_20K/images/ADE_train_00000002.jpg")
    gt = misc.imread("../ADE_20K/annotations/ADE_train_00000002.png")
    gt = (np.arange(NUM_CLASS) == gt[:,:,None] - 1)

    data, label = build_data_and_label(img, gt)
    
    data = data[0]
    label = label[0]
    label = np.transpose(label, (2,0,1))
    label = np.argmax(label, axis=0) + 1

    misc.imsave("data.png", data)
    misc.imsave("label.png", label)

