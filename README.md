# Keras implementation of [PSPNet(caffe)](https://github.com/hszhao/PSPNet)

Implemented Architecture of Pyramid Scene Parsing Network in Keras.

Converted trained weights are needed to run the network.

Weights of the original caffemodel can be converted with weight_converter.py as follows:

```bash
python weight_converter.py <path to .prototxt> <path to .caffemodel>
```

Running this needs the compiled original PSPNet caffe code and pycaffe.
Already converted weights can be downloaded here:

[pspnet50_ade20k.npy](https://www.dropbox.com/s/ms8afun494dlh1t/pspnet50_ade20k.npy?dl=0)
[pspnet101_cityscapes.npy](https://www.dropbox.com/s/b21j6hi6qql90l0/pspnet101_cityscapes.npy?dl=0)
[pspnet101_voc2012.npy](https://www.dropbox.com/s/xkjmghsbn6sfj9k/pspnet101_voc2012.npy?dl=0)

npy weights should be placed in the directory weights/npy.

The interpolation layer is implemented as custom layer "Interp"

## Keras result:
![Original](example_images/ade20k.jpg)
![New](example_results/ade20k_seg.jpg)
![New](example_results/ade20k_seg_blended.jpg)
![New](example_results/ade20k_probs.jpg)

![Original](example_images/cityscapes.png)
![New](example_results/cityscapes_seg.jpg)
![New](example_results/cityscapes_seg_blended.jpg)
![New](example_results/cityscapes_probs.jpg)

![Original](example_images/pascal_voc.jpg)
![New](example_results/pascal_voc_seg.jpg)
![New](example_results/pascal_voc_seg_blended.jpg)
![New](example_results/pascal_voc_probs.jpg)

## Pycaffe result:
![Pycaffe results](example_results/ade20k_seg_pycaffe.jpg)

## Dependencies:


1. Tensorflow (-gpu)
2. Keras
3. numpy
4. scipy
4. pycaffe(PSPNet)(optional for converting the weights)

```bash
pip install -r requirements.txt --upgrade
```

## Usage:

```bash
python pspnet.py
python pspnet.py -m pspnet101_cityscapes -i example_images/cityscapes.png -o example_results/cityscapes.jpg
python pspnet.py -m pspnet101_voc2012 -i example_images/pascal_voc.jpg -o example_results/pascal_voc.jpg
```
