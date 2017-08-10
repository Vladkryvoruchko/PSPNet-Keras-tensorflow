import os
import time
import argparse
import numpy as np
from scipy import misc

from keras.applications.resnet50 import ResNet50

from keras.layers import Input
from keras.optimizers import SGD

from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from data_generator import DiscGenerator
from datasource import DataSource
import utils

class ResnetDiscriminator:

    def __init__(self, ckpt=None):
        if ckpt is None:
            self.model = self.build_model()
        else:
            self.model = load_model(ckpt)

    def build_model(self):
        inp = Input((473,473,4))
        model = ResNet50(input_tensor=inp, weights=None, classes=2)
        sgd = SGD(lr=1e-3, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                        loss="categorical_crossentropy",
                        metrics=['accuracy'])
        return model

    def train(self, datasource, initial_epoch=0):
        path = "checkpoints/{}".format("discriminator")
        fn = "weights.{epoch:02d}-{loss:.4f}.hdf5"
        filepath = os.path.join(path, fn)
        checkpoint = ModelCheckpoint(filepath, monitor='loss')
        callbacks_list = [checkpoint]

        self.model.fit_generator(DiscGenerator(datasource), 1000, epochs=100, callbacks=callbacks_list,
            verbose=1, workers=6, initial_epoch=initial_epoch)

    def predict(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, help='Checkpoint')
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    project = "local"
    config = utils.get_config(project)
    datasource = DataSource(config, random=True)

    sess = tf.Session()
    K.set_session(sess)
    with sess.as_default():
        disc = ResnetDiscriminator()
        disc.train(datasource)
