import os
import sys
import numpy as np

import caffe

# Not needed because Tensorflow and Caffe do convolution the same way
# Needed for conversion to Theano
def rot90(W):
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W[i, j] = np.rot90(W[i, j], 2)
    return W

weights = {}
net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
for k,v in net.params.items():
	print "Layer %s, has %d params." % (k, len(v))
	if len(v) == 1:
		W = v[0].data[...]
		W = np.transpose(W, (2,3,1,0))
		weights[k] = {"weights": W}
	elif len(v) == 2:
		W = v[0].data[...]
		W = np.transpose(W, (2,3,1,0))
		b = v[1].data[...]
		weights[k] = {"weights": W, "biases": b}
	elif len(v) == 4:
		k = k.replace('/', '_')
		mean = v[0].data[...]
		variance = v[1].data[...]
		scale = v[2].data[...]
		offset = v[3].data[...]
		weights[k] = {"mean": mean, "variance": variance, "scale": scale, "offset": offset}
	else:
		print "Undefined layer"
		exit()

arr = np.asarray(weights)
np.save("pspnet50_ade20k.npy", arr)


