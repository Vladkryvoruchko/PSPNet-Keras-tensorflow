import os
import sys
import argparse
import numpy as np

from keras import backend as K
import tensorflow as tf

from pspnet_gan import PSPNetGAN
from datasource import DataSource
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--id', default="0")
parser.add_argument('--g_weights', type=str, help='Generator checkpoint')
parser.add_argument('--d_weights', type=str, help='Discriminator checkpoint')
parser.add_argument('--epoch', type=int, default=0)
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.id

project = "ade20k"
config = utils.get_config(project)
datasource = DataSource(config, random=True)


sess = tf.Session()
K.set_session(sess)
with sess.as_default():

    pspnet_gan = PSPNetGAN()
    pspnet_gan.load_weights(args.g_weights, args.d_weights)

    pspnet.train(datasource, initial_epoch=args.epoch)


