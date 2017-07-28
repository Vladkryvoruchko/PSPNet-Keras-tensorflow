import os
import sys
import time
import argparse
import numpy as np
from scipy import misc

from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import tensorflow as tf

import layers_builder as layers
import image_processor
from datasource import DataSource
from prefetcher import PreFetcher

WEIGHTS = 'pspnet50_ade20k.npy'

class PSPNet:

    def __init__(self, datasource, mode="softmax"):
        self.mode = mode

        if self.mode == "softmax":
            self.model = layers.build_pspnet()
        elif self.mode == "sigmoid":
            self.model = layers.build_pspnet_sigmoid()

        #set_weights(self.model)
        self.datasource = datasource
        self.prefetcher = PreFetcher(datasource)

    def generator(self):
        while True:
            im = self.datasource.next_im()
            print im
            t = time.time()
            img = self.datasource.get_image(im)
            gt = self.datasource.get_ground_truth(im)
            data,label = image_processor.build_data_and_label(img, gt)
            print time.time() - t
            print data.shape, label.shape
            yield (data,label)

    def train(self):
        path = "checkpoints/{}".format(self.mode)
        fn = "weights.{epoch:02d}-{val_loss:.2f}.hdf5"
        filepath = os.path.join(path, fn)
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss')
        callbacks_list = [checkpoint]
        
        adam = Adam(lr=1e-5)
        self.model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
        self.model.fit_generator(self.generator(), 1000, epochs=100, verbose=2, callbacks=callbacks_list, use_multiprocessing=False)

    def predict_sliding_window(self, img):
        patches = image_processor.build_sliding_window(img)
        print patches.shape
        crop_probs = []
        for patch in patches:
            crop_prob = self.feed_forward(patch)
            crop_probs.append(crop_prob)
        probs = image_processor.post_process_sliding_window(img, crop_probs)
        return probs

    def predict(self, img):
        img = misc.imresize(img, (473, 473))
        img = image_processor.preprocess(img)
        probs = self.feed_forward(img)
        return probs

    def feed_forward(self, data):
        '''
        Input must be 473x473x3 in RGB
        Output is 150x473x473
        '''
        assert data.shape == (473,473,3)
        data = data[np.newaxis,:,:,:]
        print array_to_str(data)

        self.debug(data)
        pred = self.model.predict(data, batch_size=1)
        return pred

    def debug(self, data):
        names = [layer.name for layer in self.model.layers]
        for name in names[-12:]:
            print_activation(self.model, name, data)

def print_activation(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print layer_name, array_to_str(io)

def array_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a), np.max(a), np.mean(a))

def set_weights(model):
    print 'Opening weights...'
    weights = np.load(WEIGHTS).item()
    print 'Loading weights...'

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
    args = parser.parse_args()

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        pspnet = PSPNet(None)

        img = misc.imread(args.input_path)
        print array_to_str(img)

        probs = pspnet.predict(img)
        print probs.shape

        cm = np.argmax(probs, axis=2) + 1
        color_cm = utils.add_color(cm)
        misc.imsave(args.output_path, color_cm)

