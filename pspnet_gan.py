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
from keras.utils import plot_model
import tensorflow as tf

from gan_builder import DCGAN
from datasource import DataSource
import image_processor
import utils

class PSPNetGAN:

    def __init__(self, mode="baseline"):
        self.mode = mode

        self.dcgan = DCGAN(disc_use_features=False)
        self.generator = self.dcgan.generator
        self.discriminator = self.dcgan.discriminator
        self.adversarial = self.dcgan.adversarial

    def train(self, datasource, train_steps=1000, epochs=100, initial_epoch=0):
        g = data_generator(datasource)
        for e in xrange(initial_epoch, epochs):
            print "Epoch {}/{}".format(e, epochs)
            loss = None
            for i in xrange(train_steps):
                t1 = time.time()
                data,label = g.next()
                t2 = time.time()
                print "Get: ", t2-t1
                
                # Train discriminator
                label_fake = self.generator.predict(data)
                g_loss = self.generator.test_on_batch(data, label)

                x = np.concatenate((label, label_fake))
                y = np.array([1,0])
                imgs = np.concatenate((data, data))

                d_loss = self.discriminator.train_on_batch({'pred': x}, {'d_out': y})
                print self.discriminator.predict({'pred': x})
                # d_loss = self.discriminator.train_on_batch({'pred': x, 'img': imgs}, {'d_out': y})

                # Train adversarial
                y = np.array([1])
                #a_loss = self.adversarial.train_on_batch({'img': data}, {'pred': label, 'd_out': y})
                a_loss = self.adversarial.train_on_batch({'img': data}, {'d_out': y})

                print "{}: [D loss: {}, acc: {}]".format(i, d_loss[0], d_loss[1])
                print "{}: [A loss: {}, acc: {}]".format(i, a_loss[0], a_loss[1])
                print "{}: [G loss: {}, acc: {}]".format(i, g_loss[0], g_loss[1])
                loss = d_loss[0]
                t3 = time.time()
                print "Train: ", t3-t2

            # Checkpoint
            fn = "weights.{}-{}.hdf5".format(e, loss)
            self.checkpoint(fn)

    def checkpoint(self, fn):
        checkpoints_dir = "checkpoints/{}/".format(self.mode)
        g_path = os.path.join(checkpoints_dir, "generator")
        d_path = os.path.join(checkpoints_dir, "discriminator")
        if not os.path.exists(g_path):
            os.makedirs(g_path)
        if not os.path.exists(d_path):
            os.makedirs(d_path)
        g_fn = os.path.join(g_path, fn)
        d_fn = os.path.join(d_path, fn)
        self.generator.save(g_fn)
        self.discriminator.save(d_fn)
        print "Checkpointed. ", g_fn
        print "Checkpointed. ", d_fn

    def load_weights(self, g_weights, d_weights):
        if g_weights is not None:
            print "Loading generator weights:", g_weights
            self.generator.load_weights(g_weights)
        if d_weights is not None:
            print "Loading discriminator weights:", d_weights
            self.discriminator.load_weights(d_weights)

    def predict_sliding_window(self, img):
        patches = image_processor.build_sliding_window(img)

        outputs = []
        for patch in patches:
            out = self.feed_forward(patch)
            outputs.append(out)
        crop_probs = np.concatenate(outputs, axis=0)
        probs = image_processor.post_process_sliding_window(img, crop_probs)
        return probs

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

        # debug(self.generator, data)
        pred = self.generator.predict(data)
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
    parser.add_argument('--g_weights', type=str, help='Generator checkpoint')
    parser.add_argument('--d_weights', type=str, help='Discriminator checkpoint')
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = misc.imread(args.input_path)
        
        pspnet_gan = PSPNetGAN()
        pspnet_gan.load_weights(args.g_weights, args.d_weights)

        probs = pspnet_gan.predict(img)
        #probs = pspnet_gan.predict_sliding_window(img)

        cm = np.argmax(probs, axis=2) + 1
        color_cm = utils.add_color(cm)
        misc.imsave(args.output_path, color_cm)

        print "GENERATOR"
        model = pspnet_gan.generator
        model.summary()
        print "DISCRIMINATOR"
        model = pspnet_gan.discriminator
        model.summary()
        print "ADVERSARIAL"
        model = pspnet_gan.adversarial
        model.summary()

        # Plot networks
        # plot_model(pspnet_gan.generator, to_file='gen.png', show_shapes=True)
        # plot_model(pspnet_gan.discriminator, to_file='disc.png', show_shapes=True)
        # plot_model(pspnet_gan.adversarial, to_file='adversarial.png', show_shapes=True)

