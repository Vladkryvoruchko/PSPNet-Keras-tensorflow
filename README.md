# Keras implementation of [PSPNet(caffe)](https://github.com/hszhao/PSPNet)

Implemented Architecture of pyramid scene parsing network in Keras

Converted trained weights needed to run the network.


Download converted weights here:
[link:pspnet.npy](https://www.dropbox.com/s/9xebhix7dbk372d/pspnet.npy?dl=0)

And place in directory with pspnet.py

Weights from caffemodel were converted with [caffe-tensorflow](https://github.com/ethereon/caffe-tensorflow), source code of converter was modified to fit batch normalization, which is annotated as 'BN' in original prototxt

Interpolation layer is implemented in code as custom layer "Interp"

## Important

This implementation is not working properly despite calculations are made without errors(output image is very very bad).
I can't figure out which causes such behavior, so help and proposals are appreciated.

Memory usage:3500Mb
Calculation speed: 1.2 sec on gtx 1080

## Dependencies:
1. Tensorflow
2. Keras
3. numpy


## Usage: 

```bash
python pspnet.py
```
this causes to start 
