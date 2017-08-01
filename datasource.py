import os
import time
import random
import h5py
import numpy as np
from scipy import misc
from keras.utils import to_categorical
import utils

NUM_CLASS = 150

class DataSource:
    def __init__(self, config, random=True):
        self.image_dir = config["images"]
        self.ground_truth_dir = config["ground_truth"]

        im_list_txt = config["im_list"]
        self.im_list = utils.open_im_list(im_list_txt)
        
        self.random = random
        if not self.random:
            self.idx = -1

    def next_im(self):
        if self.random:
            idx = random.randint(0,len(self.im_list)-1)
            return self.im_list[idx]
        else:
            self.idx += 1
            if self.idx == len(self.im_list):
                self.idx = 0
            return self.im_list[self.idx]

    def get_image(self, im):
        img_path = os.path.join(self.image_dir, im)
        img = misc.imread(img_path)
        if img.ndim != 3:
            img = np.stack((img,img,img), axis=2)
        return img

    def get_ground_truth(self, im):
        gt_path = os.path.join(self.ground_truth_dir, im.replace('.jpg', '.png'))
        gt = misc.imread(gt_path)
        gt = (np.arange(NUM_CLASS) == gt[:,:,None] - 1)
        
        #gt = gt.astype(int)
        #for i in xrange(NUM_CLASS):
        #    if np.sum(gt[:,:,i]) == 0:
        #        gt[:,:,i] = -1
        #print np.min(gt), np.max(gt), np.mean(gt)
        return gt

