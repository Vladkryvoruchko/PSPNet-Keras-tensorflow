from keras import backend as K
from PIL import Image

import layers_builder as pspnet
import tensorflow as tf
import numpy as np
import drawImage
import argparse
import time



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

			# mean *= scale
			# variance *= scale
			
			# model.get_layer(layer.name).set_weights([mean, variance,
			# 										scale, offset])
			model.get_layer(layer.name).set_weights([scale, offset,
													mean, variance])
			# model.get_layer(layer.name).set_weights([scale, offset,
													# mean, variance])

		elif layer.name[:4] == 'conv' and not layer.name[-4:] == 'relu':
			print layer.name
			try:
				weight = weights[layer.name]['weights']
				model.get_layer(layer.name).set_weights([weight])
			except Exception as err:
				biases = weights[layer.name]['biases']
				model.get_layer(layer.name).set_weights([weight, biases])

	print 'weights set finish'
	return model


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

	model = pspnet.build_pspnet()

	sess = tf.Session()
	K.set_session(sess)

	with sess.as_default():
		#Load weights into variable
		npy_weights = load_weights()
		#Set weights to each laye by name
		model = set_weights(model, npy_weights)

		#Load image, resize and paste into 4D tensor
		image = Image.open(settings.input_path)
		im = image.resize((473, 473))
		input_ = np.array(im, dtype=np.float32)
		input_ = input_[:,:,::-1]
		input_ -= np.array((mean_b, mean_g, mean_r))
		data = np.zeros([1,473,473,3])

		data[0] = input_

		#predict
		
		startForward = time.time()
		pred = model.predict(data, batch_size=1, verbose=0)
		finishForward = (time.time() - startForward)
		print "Time used: %f" % finishForward
		# pred = np.transpose(pred[0], (2, 1, 0))
		print np.shape(pred)
		pred = pred[0]
		predicted_classes = np.argmax(pred, axis=2)

		proto = 'utils/model/pspnet.prototxt'
		weights = 'utils/model/pspnet.caffemodel'
		colors = 'utils/colorization/color150.mat'
		objects = 'utils/colorization/objectName150.mat'


		im_Width = predicted_classes.shape[0]
		im_Height = predicted_classes.shape[1]
		draw = drawImage.BaseDraw(colors, objects,
							image, (im_Width, im_Height),
							predicted_classes)
		simpleSegmentImage = draw.drawSimpleSegment();
		simpleSegmentImage.save(settings.output_path,"JPEG")


