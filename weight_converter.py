import caffe
import numpy as np
import os, sys

weights = {}
net = caffe.Net(sys.argv[1], sys.argv[2], caffe.TEST)
for k,v in net.params.items():
	print "Layer %s, has %d params." % (k, len(v))
	if len(v) == 1:
		weights[k] = {"weights": np.transpose(v[0].data[...], (2,3,1,0))}
	elif len(v) == 2:
		weights[k] = {"weights": np.transpose(v[0].data[...], (2,3,1,0)), "biases": v[1].data[...]}
	elif len(v) == 4:
		weights[k.replace('/', '_')] = {"scale": v[0].data[...], "offset": v[1].data[...], "mean": v[2].data[...], "variance": v[3].data[...]}
	else:
		print "Undefined layer"
		exit()

arr = np.asarray(weights)
np.save("pspnet50_ade20k.npy", arr)