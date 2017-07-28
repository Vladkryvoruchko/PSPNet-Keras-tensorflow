import os
import argparse
import numpy as np
import h5py
import random
from scipy import misc

from keras import backend as K
import tensorflow as tf

from pspnet import PSPNet
from datasource import DataSource
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--id', default=0,type=int)
parser.add_argument("-p", required=True, help="Project name")
parser.add_argument("--mode", required=True, help="softmax, sigmoid, etc")
parser.add_argument("--checkpoint", help="Checkpoint .hdf5")
args = parser.parse_args()

project = args.p
mode = args.mode
checkpoint = args.checkpoint

root_result = "predictions/original/"
if checkpoint is not None:
    root_result = "predictions/{}/{}/".format(mode, os.path.basename(checkpoint).split('-')[0])
print "Outputting to ", root_result

root_mask = os.path.join(root_result, 'category_mask')
root_prob = os.path.join(root_result, 'prob_mask')
root_maxprob = os.path.join(root_result, 'max_prob')
root_allprob = os.path.join(root_result, 'all_prob')

sess = tf.Session()
K.set_session(sess)
with sess.as_default():
    config = utils.get_config(project)
    datasource = DataSource(config, random=False)
    pspnet = PSPNet(datasource, mode=mode, ckpt=checkpoint)

    random.seed(3)
    for im in datasource.im_list:
        print im

        fn_maxprob = os.path.join(root_maxprob, im.replace('.jpg', '.h5'))
        fn_mask = os.path.join(root_mask, im.replace('.jpg', '.png'))
        fn_prob = os.path.join(root_prob, im)
        fn_allprob = os.path.join(root_allprob, im.replace('.jpg', '.h5'))

        if os.path.exists(fn_maxprob):
            print "Already done."
            continue

        # make paths if not exist
        if not os.path.exists(os.path.dirname(fn_maxprob)):
            os.makedirs(os.path.dirname(fn_maxprob))
        if not os.path.exists(os.path.dirname(fn_mask)):
            os.makedirs(os.path.dirname(fn_mask))
        if not os.path.exists(os.path.dirname(fn_prob)):
            os.makedirs(os.path.dirname(fn_prob))
        if not os.path.exists(os.path.dirname(fn_allprob)):
            os.makedirs(os.path.dirname(fn_allprob))

        img = datasource.get_image(im)
        probs = pspnet.predict_sliding_window(img)
        # probs is 150 x h x w

        # calculate output
        pred_mask = np.argmax(probs, axis=2) + 1
        prob_mask = np.max(probs, axis=2)
        max_prob = np.max(probs, axis=(0,1))
        all_prob = probs

        # write to file
        misc.imsave(fn_mask, pred_mask.astype('uint8'))
        misc.imsave(fn_prob, (prob_mask*255).astype('uint8'))
        with h5py.File(fn_maxprob, 'w') as f:
            f.create_dataset('maxprob', data=max_prob)
        with h5py.File(fn_allprob, 'w') as f:
            f.create_dataset('allprob', data=all_prob)
