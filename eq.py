import h5py
import numpy as np

k = None
c = None
with h5py.File("keras.h5", 'r') as f:
    k = f['a'][:]
with h5py.File("caffe.h5", 'r') as f:
    c = f['a'][:]

k = np.transpose(k[0], (2,0,1))

print k.shape, np.min(k), np.max(k), np.mean(k)
print c.shape, np.min(c), np.max(c), np.mean(c)

k = k[2048:2048+512]
c = c[4096-512:4096]
# print k
# print c
diff = np.abs(k - c)
# sig = diff > 2
# print np.sum(sig)
# i = np.argmax(diff)
# i = np.unravel_index(i, diff.shape)
# print k[i], c[i]
print diff.shape, np.max(diff), np.mean(diff)