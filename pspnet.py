import os
import sys
import time
import numpy as np
from scipy import misc

from keras import backend as K
import tensorflow as tf

import layers_builder as layers
from datasource import DataSource
from image_processor import ImageProcessor

WEIGHTS = 'pspnet50_ade20k.npy'

class PSPNet:

    def __init__(self, datasource):
        self.model = layers.build_pspnet()
        set_weights(self.model)

        self.image_processor = ImageProcessor()
        # self.prefetcher = PreFetcher(datasource, image_processor)

    # def train(self):
    #     model.fit_generator(self.prefetcher.fetch_batch(), samples_per_epoch=20000, nb_epoch=10)

    def predict_sliding_window(self, img):
        patches = self.image_processor.sliding_window(img)
        print patches.shape
        crop_probs = []
        for patch in patches:
            crop_prob = self.feed_forward(patch)
            crop_probs.append(crop_prob)
        probs = self.image_processor.post_process_sliding_window(img, crop_probs)
        return probs

    def feed_forward(self, data):
        '''
        Input must be 473x473x3 in RGB
        Output is 150x473x473
        '''
        assert data.shape == (473,473,3)
        data = data[np.newaxis,:,:,:]

        self.debug(data)
        pred = model.predict(data, batch_size=1)
        return pred

    def debug(self, data):
        names = [layer.name for layer in self.model.layers]
        for name in names[-20:]:
            print_activation(self.model, name, data)

def print_activation(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print layer_name, np_to_str(io)

def np_to_str(a):
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
        img = misc.imresize(img, (473, 473))

        probs = pspnet.predict_sliding_window(img)
        print probs.shape

        cm = np.argmax(probs, axis=2) + 1
        color_cm = utils.add_color(cm)
        misc.imsave(args.output_path, color_cm)

