from keras import backend as K
from keras.models import Model

import layers_builder as pspnet
import tensorflow as tf
import numpy as np
import drawImage
import argparse
import time

from scipy import misc
import utils



def load_weights():
    w = np.load('pspnet50_ade20k.npy').item()
    return w


def set_weights(model, weights):
    print 'weights set start'
    for layer in model.layers:
        if layer.name[:4] == 'conv' and layer.name[-2:] == 'bn':
            print layer.name
            scale = weights[layer.name]['scale'].reshape(-1)

            offset = weights[layer.name]['offset'].reshape(-1)
            mean = weights[layer.name]['mean'].reshape(-1)
            variance = weights[layer.name]['variance'].reshape(-1)

            print "mean", np_to_str(mean)
            print "variance", np_to_str(variance)
            print "scale", np_to_str(scale)
            print "offset", np_to_str(offset)

            # mean *= scale
            # variance *= scale
            
            model.get_layer(layer.name).set_weights([mean, variance, scale, offset])
            #model.get_layer(layer.name).set_weights([scale, offset, mean, variance])
            # model.get_layer(layer.name).set_weights([scale, offset,
                                                    # mean, variance])

        elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
            print layer.name
            try:
                weight = weights[layer.name]['weights']
                print "weights", np_to_str(weight)
                model.get_layer(layer.name).set_weights([weight])
            except Exception as err:
                biases = weights[layer.name]['biases']
                print "biases", np_to_str(biases)
                model.get_layer(layer.name).set_weights([weight, biases])
        # else:
        #   print layer.name, "missing"

    print 'weights set finish'
    return model

def print_activation(model, layer_name, data):
    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=model.get_layer(layer_name).output)
    io = intermediate_layer_model.predict(data)
    print layer_name, np_to_str(io)

def np_to_str(a):
    return "{} {} {} {} {}".format(a.dtype, a.shape, np.min(a), np.max(a), np.mean(a))


if __name__ == "__main__":

    settings = None
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-path', type=str, default='', 
                        required=True, help='Path the input image')
    parser.add_argument('--output-path', type=str, default='',
                        required=True, help='Path to output')

    settings, unparsed = parser.parse_known_args()
    mean_r = 123.68
    mean_g = 116.779
    mean_b = 103.939
    DATA_MEAN = np.array([[[123.68, 116.779, 103.939]]])

    #Load image, resize and paste into 4D tensor
    img = misc.imread(settings.input_path)
    img = misc.imresize(img, (473, 473))
    print np_to_str(img)
    img = img - DATA_MEAN
    img = img.astype('float32')
    img = img[:,:,::-1]

    data = img[np.newaxis, ...]
    print np_to_str(data)

    model = pspnet.build_pspnet()

    sess = tf.Session()
    K.set_session(sess)

    with sess.as_default():
        #Load weights into variable
        npy_weights = load_weights()
        #Set weights to each laye by name
        model = set_weights(model, npy_weights)


        for layer in model.layers:
            name = layer.name
            if "activation_57" in name or "activation_58" in name or "lambda" in name or "concatenate_1" in name or "conv5_4" in name:
                print_activation(model, name, data)

        #predict
        
        startForward = time.time()
        pred = model.predict(data, batch_size=1, verbose=0)
        finishForward = (time.time() - startForward)
        print "Time used: %f" % finishForward
        # pred = np.transpose(pred[0], (2, 1, 0))
        print np.shape(pred)
        pred = pred[0]
        predicted_classes = np.argmax(pred, axis=2) + 1

        color = utils.add_color(predicted_classes)
        misc.imsave(settings.output_path, color)


        # proto = 'utils/model/pspnet.prototxt'
        # weights = 'utils/model/pspnet.caffemodel'
        # colors = 'utils/colorization/color150.mat'
        # objects = 'utils/colorization/objectName150.mat'


        # im_Width = predicted_classes.shape[0]
        # im_Height = predicted_classes.shape[1]
        # draw = drawImage.BaseDraw(colors, objects,
        #                   image, (im_Width, im_Height),
        #                   predicted_classes)
        # simpleSegmentImage = draw.drawSimpleSegment();
        # simpleSegmentImage.save(settings.output_path,"JPEG")


