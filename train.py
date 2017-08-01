import os
import sys
import argparse
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from pspnet import PSPNet
from datasource import DataSource
import utils

def get_latest_checkpoint(checkpoint_dir):
    # weights.00-1.52.hdf5
    latest_i = -1
    latest_fn = ""
    for fn in os.listdir(checkpoint_dir):
        split0 = fn.split('-')[0]
        i = int(split0.split('.')[1])
        if i > latest_i:
            latest_i = i
            latest_fn = fn

    if latest_i == -1:
        raise Exception("No checkpoint found.")
    return os.path.join(checkpoint_dir, latest_fn), latest_i+1


parser = argparse.ArgumentParser()
parser.add_argument('--id', default="0")
parser.add_argument("--mode", required=True, help="softmax, sigmoid, etc")
parser.add_argument('--resume', action='store_true', default=False)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.id

project = "ade20k"
mode = args.mode
config = utils.get_config(project)
datasource = DataSource(config, random=True)

checkpoint = None
epoch = 0
if args.resume:
    checkpoint_dir = "checkpoints/{}/".format(mode)
    checkpoint,epoch = get_latest_checkpoint(checkpoint_dir)


sess = tf.Session()
K.set_session(sess)
with sess.as_default():
    pspnet = PSPNet(datasource, mode=mode, ckpt=checkpoint)
    pspnet.train(initial_epoch=epoch)
