from keras import backend as K
from keras.models import Model

import tensorflow as tf
import numpy as np
import argparse
import time

from scipy import misc
import utils

if __name__ == "__main__":

    settings = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='', 
                        required=True, help='Path the input image')
    parser.add_argument('--output-path', type=str, default='',
                        required=True, help='Path to output')

    settings, unparsed = parser.parse_args()

    data = img[np.newaxis, ...]

    model = pspnet.build_pspnet()

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():

        model = set_weights(model)
        names = [layer.name for layer in model.layers]
        for name in names[-20:]:
            print_activation(model, name, data)

        #predict
        pred = model.predict(data, batch_size=1, verbose=0)
        print np.shape(pred)
        pred = pred[0]
        predicted_classes = np.argmax(pred, axis=2) + 1

        color = utils.add_color(predicted_classes)
        misc.imsave(settings.output_path, color)

