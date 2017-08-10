import random
import numpy as np
import itertools
from scipy import misc, ndimage
import cv2

INPUT_SIZE = 473
NUM_CLASS = 150

stride_rate = 0.3

def crop_array(a, box):
    h,w,c = a.shape
    sh,eh,sw,ew = box
    crop = np.zeros((INPUT_SIZE,INPUT_SIZE, c), dtype=int)
    crop[0:eh-sh,0:ew-sw] = img[sh:eh,sw:ew]
    return crop

def random_crop(img):
    h,w,_ = img.shape

    sh = 0
    sw = 0
    if h > INPUT_SIZE:
        sh = random.randint(0,h-INPUT_SIZE)
    if w > INPUT_SIZE:
        sw = random.randint(0,w-INPUT_SIZE)
    eh = min(h,sh + INPUT_SIZE)
    ew = min(w,sw + INPUT_SIZE)
    box = (sh,eh,sw,ew)
    return box

def crop_sliding_window(img):
    h,w,_ = img.shape
    crop_boxes = sw_crop_boxes(h,w)
    n = len(crop_boxes)

    crops = np.zeros((n,INPUT_SIZE,INPUT_SIZE,3))
    for i in xrange(n):
        box = crop_boxes[i]
        crops[i] = crop_image(img, box)
    return crops

def assemble_probs(shape, crop_probs):
    h,w = shape[:2]
    probs = np.zeros((h, w, NUM_CLASS), dtype=np.float32)
    cnts = np.zeros((h,w,1))

    boxes = sw_crop_boxes(h,w)
    n = len(boxes)
    for i in xrange(n):
        sh,eh,sw,ew = boxes[i]
        crop_prob = crop_probs[i]

        probs[sh:eh,sw:ew,:] += crop_prob[0:eh-sh,0:ew-sw,:]
        cnts[sh:eh,sw:ew,0] += 1

    assert cnts.min()>=1
    probs /= cnts
    assert (probs.min()>=0 and probs.max()<=1), '%f,%f'%(probs.min(),probs.max())
    return probs

def sw_crop_boxes(h,w):
    '''
    Sliding window crop box locations
    '''
    # Get top-left corners
    stride = INPUT_SIZE * stride_rate
    hs_upper = max(1,h-(INPUT_SIZE-stride))
    ws_upper = max(1,w-(INPUT_SIZE-stride))
    hs = np.arange(0,hs_upper,stride, dtype=int)
    ws = np.arange(0,ws_upper,stride, dtype=int)
    crop_locs = list(itertools.product(hs,ws))

    boxes = []
    for loc in crop_locs:
        sh,sw = loc
        eh = min(h, sh + INPUT_SIZE)
        ew = min(w, sw + INPUT_SIZE)
        box = (sh,eh,sw,ew)
        boxes.append(box)
    return boxes

def scale_maxside(a, maxside=512):
    h,w = a.shape[:2]
    long_side = max(h, w)
    r = 1.*maxside/long_side # Make long_side == scale_size

    h_t = h*r
    w_t = w*r
    return scale(a, (h_t,w_t))

def scale(a, shape):
    h_t,w_t = shape[:2]
    h,w = a.shape[:2]
    r_h = 1.*h_t/h
    r_w = 1.*w_t/w

    if np.ndim(a) == 3 and a.shape[2] == 3:
        # Image, use bilinear
        return ndimage.zoom(a, (r_h,r_w,1.), order=1, prefilter=False)
    else:
        # Ground truth, use nearest
        if np.ndim(a) == 2:
            return ndimage.zoom(a, (r_h,r_w), order=0, prefilter=False)
        else:
            return ndimage.zoom(a, (r_h,r_w,1.), order=0, prefilter=False)
