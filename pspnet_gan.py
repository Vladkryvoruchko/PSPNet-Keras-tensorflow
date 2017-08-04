import os
import sys
import time
import h5py
import argparse
import numpy as np
from scipy import misc

from keras import backend as K
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

from gan_builder import DCGAN
from datasource import DataSource
import image_processor
import utils

class PSPNetGAN:

    def __init__(self):
        self.dcgan = DCGAN()
        self.generator = self.dcgan.generator
        self.discriminator = self.dcgan.discriminator
        self.adversarial = self.dcgan.adversarial

    def set_weights(self, weights):
        print "Loading from", weights
        if '.h5' in weights or '.hdf5' in weights:
            self.model.load_weights(weights)
        else:
            raise Exception('Weight file format not recognized.')

    def train(self, datasource, train_steps=1000, epochs=100, initial_epoch=0):
        g = data_generator(datasource)
        for e in xrange(initial_epoch, epochs):
            print "Epoch {}/{}".format(e, epochs)
            for i in xrange(train_steps):
                data,label = g.next()

                # Train discriminator
                label_fake = self.generator.predict(data)
                x = np.concatenate((label, label_fake))
                y = np.array([1,0])
                imgs = np.concatenate((data, data))

                d_loss = self.discriminator.train_on_batch({'pred': x}, {'d_out': y})
                # d_loss = self.discriminator.train_on_batch({'pred': x, 'img': imgs}, {'d_out': y})

                # Train adversarial
                y = np.array([1])
                a_loss = self.adversarial.train_on_batch({'img': data}, {'pred': label, 'd_out': y})

                print "{}: [D loss: {}, acc: {}]".format(i, d_loss[0], d_loss[1])
                print "{}: [A loss: {}, acc: {}]".format(log_mesg, a_loss[0], a_loss[1])

            # Checkpoint
            # path = "checkpoints/{}".format(self.mode)
            # fn = "weights.{epoch:02d}-{loss:.4f}.hdf5"
            # filepath = os.path.join(path, fn)
            # print "Checkpoint:", filepath

    def predict(self, img):
        img = misc.imresize(img, (473, 473))
        img = image_processor.preprocess(img)
        probs = self.feed_forward(img)
        return probs[0]

    def feed_forward(self, data):
        '''
        Input must be 473x473x3 in RGB
        Output is 150x473x473
        '''
        assert data.shape == (473,473,3)
        data = data[np.newaxis,:,:,:]

        # debug(self.model, data)
        pred = self.model.predict(data)
        return pred

def data_generator(datasource):
    while True:
        im = datasource.next_im()
        #print im
        #t = time.time()
        img = datasource.get_image(im)
        gt = datasource.get_ground_truth(im)
        data,label = image_processor.build_data_and_label(img, gt)
        #print time.time() - t
        yield (data,label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default='', required=True, help='Path the input image')
    parser.add_argument('--output_path', type=str, default='', required=True, help='Path to output')
    parser.add_argument('--checkpoint', type=str, help='Checkpoint')
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = misc.imread(args.input_path)
        
        pspnet = None
        if args.checkpoint is None:
            pspnet = PSPNet("softmax")
            pspnet.load_default_weights()
        else:
            pspnet = PSPNet(None, ckpt=args.checkpoint)

        probs = pspnet.predict(img)
        #probs = pspnet.predict_sliding_window(img)

        cm = np.argmax(probs, axis=2) + 1
        color_cm = utils.add_color(cm)
        misc.imsave(args.output_path, color_cm)

