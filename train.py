import os
import sys
import numpy as np

from keras import backend as K
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from pspnet import PSPNet
from datasource import DataSource
import utils

project = "ade20k"
config = utils.get_config(project)
datasource = DataSource(config, random=True)

sess = tf.Session()
K.set_session(sess)

with sess.as_default():
    pspnet = PSPNet(datasource)
    pspnet.train()