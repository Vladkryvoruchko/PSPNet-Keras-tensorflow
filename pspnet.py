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

import image_processor
import utils
import layers_builder as layers
from data_generator import DataGenerator

class PSPNet:

    def __init__(self, mode, ckpt=None):
        self.mode = mode
        if ckpt is not None:
            print "Loading from checkpoint:", ckpt
            self.model = load_model(ckpt, custom_objects={'Interp': layers.Interp,
                                                            'tf': tf})
        else:
            print "Building model"
            # Build model
            if "softmax" in self.mode:
                self.model = layers.build_pspnet(activation="softmax")
            elif "sigmoid" in self.mode:
                self.model = layers.build_pspnet(activation="sigmoid")

    def set_weights(self, weights):
        print "Loading from", weights
        if '.h5' in weights or '.hdf5' in weights:
            self.model.load_weights(weights)
        elif '.npy' in weights:
            set_npy_weights(self.model, weights)
        else:
            raise Exception('Weight file format not recognized.')

    def load_default_weights(self):
        WEIGHTS = 'pspnet50_ade20k.npy'
        self.set_weights(WEIGHTS)

    def train(self, datasource, initial_epoch=0):
        path = "checkpoints/{}".format(self.mode)
        fn = "weights.{epoch:02d}-{loss:.4f}.hdf5"
        filepath = os.path.join(path, fn)
        checkpoint = ModelCheckpoint(filepath, monitor='loss')
        callbacks_list = [checkpoint]

        self.model.fit_generator(DataGenerator(datasource), 1000, epochs=100, callbacks=callbacks_list,
                 verbose=1, workers=6, initial_epoch=initial_epoch)

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

        # debug(self.model, data)
        pred = self.model.predict(data)
        return pred

def output(x,y,prefix="",slice_y=False):
    x = x[0]
    y = y[0]
    misc.imsave(prefix+"x.png", x)
    cm = np.argmax(y, axis=2) + 1
    cm = utils.add_color(cm)
    misc.imsave(prefix+"cm.png", cm)
    if slice_y:
        for i in xrange(150):
            if np.max(y[:,:,i]) > 0.1:
                misc.imsave(prefix+str(i)+".png", y[:,:,i])
    

def debug(model, data):
    names = [layer.name for layer in model.layers]
    for name in names[:]:
        print_activation(model, name, data)

def print_activation(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print layer_name, array_to_str(io)
    #if layer_name == "concatenate_1":
    #    print "Saving", layer_name
    #    with h5py.File("keras.h5", 'w') as f:
    #        f.create_dataset('a', data=io)
def array_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a), np.max(a), np.mean(a))

def set_npy_weights(model, npy_weights):
    weights = np.load(npy_weights).item()

    for layer in model.layers:
        print layer.name
        if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
            mean = weights[layer.name]['mean'].reshape(-1)
            variance = weights[layer.name]['variance'].reshape(-1)
            scale = weights[layer.name]['scale'].reshape(-1)
            offset = weights[layer.name]['offset'].reshape(-1)
            
            model.get_layer(layer.name).set_weights([mean, variance, scale, offset])

        elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
            try:
                weight = weights[layer.name]['weights']
                model.get_layer(layer.name).set_weights([weight])
            except Exception as err:
                biases = weights[layer.name]['biases']
                model.get_layer(layer.name).set_weights([weight, biases])
    print 'Finished.'
    return model

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

