from __future__ import print_function
from math import ceil
from keras import layers
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Lambda
from keras.layers.merge import Concatenate, Add
from keras.models import Model
from keras.optimizers import SGD
from keras.backend import tf as ktf
from .keras_layers import AdaptivePooling2D, Interp

import tensorflow as tf


def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = Conv2D(64 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(64 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad, pad))(prev)
    prev = Conv2D(64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                  name=names[2], use_bias=False)(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[4],
                  use_bias=False)(prev)
    prev = BN(name=names[5])(prev)
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = Conv2D(256 * level, (1, 1), strides=(1, 1), name=names[0],
                      use_bias=False)(prev)
    elif modify_stride is True:
        prev = Conv2D(256 * level, (1, 1), strides=(2, 2), name=names[0],
                      use_bias=False)(prev)

    prev = BN(name=names[1])(prev)
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = Activation('relu')(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = Add()([block_1, block_2])
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = Add()([block_1, block_2])
    return added


def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                  use_bias=False)(inp)  # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                  use_bias=False)(relu1)  # "conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  # "conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                  use_bias=False)(relu1)  # "conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)  # "conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)  # "conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3, 3), padding='same',
                       strides=(2, 2))(relu1)  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = Activation('relu')(res)
    return res


def interp_block(prev_layer, level, output_size):
    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]

    prev_layer = AdaptivePooling2D(level, mode='avg')(prev_layer)
    prev_layer = Conv2D(512, (1, 1), strides=(1, 1), name=names[0],
                        use_bias=False)(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)

    prev_layer = Interp(output_size)(prev_layer)
    return prev_layer


def build_pyramid_pooling_module(res):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation

    shape = ktf.shape(res)
    output_size = (shape[1], shape[2])
    interp_block1 = interp_block(res, 1, output_size)
    interp_block2 = interp_block(res, 2, output_size)
    interp_block3 = interp_block(res, 3, output_size)
    interp_block6 = interp_block(res, 6, output_size)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    first = Interp(output_size)(res)

    res = Concatenate()([first,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1])
    return res


def build_pspnet(nb_classes, resnet_layers, input_shape, activation='softmax'):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i expecting inputs of shape %s predicting %i classes" % (
        resnet_layers, input_shape, nb_classes))

    inp = Input((None, None, 3))
    input_shape = ktf.shape(inp)
    res = ResNet(inp, layers=resnet_layers)

    psp = build_pyramid_pooling_module(res)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
               use_bias=False)(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(nb_classes, (1, 1), strides=(1, 1), name="conv6")(x)

    x = Interp([input_shape[1], input_shape[2]])(x)
    x = Activation(activation)(x)

    model = Model(inputs=inp, outputs=x)

    return model
