import os
import argparse
import numpy as np
from scipy import misc, ndimage

from keras import backend as K
import tensorflow as tf

import layers_builder as layers
import utils

WEIGHTS = 'pspnet50_ade20k.npy'
DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]]) # RGB

class PSPNet:

    def __init__(self):
        self.model = layers.build_pspnet()
        set_npy_weights(self.model, WEIGHTS)

    def predict(self, img):
        '''
        Arguments:
            img: must be 473x473x3
        '''
        h_ori,w_ori = img.shape[:2]

        # Preprocess
        img = misc.imresize(img, (473, 473))
        img = img - DATA_MEAN
        img = img[:,:,::-1] # RGB => BGR
        img = img.astype('float32')

        probs = self.feed_forward(img)
        h,w = probs.shape[:2]
        probs = ndimage.zoom(probs, (1.*h_ori/h,1.*w_ori/w,1.), order=1, prefilter=False)
        return probs

    def predict_sliding_window(self, img):
        pass

    def feed_forward(self, data):
        assert data.shape == (473,473,3)
        data = data[np.newaxis,:,:,:]

        # utils.debug(self.model, data)
        pred = self.model.predict(data)
        return pred[0]

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
    parser.add_argument('--id', default="0")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.id

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        img = misc.imread(args.input_path)
        
        pspnet = PSPNet()
        probs = pspnet.predict(img)

        cm = np.argmax(probs, axis=2) + 1
        pm = np.max(probs, axis=2)
        color_cm = utils.add_color(cm)
        misc.imsave(args.output_path, color_cm)
        misc.imsave("probs.jpg", pm)

