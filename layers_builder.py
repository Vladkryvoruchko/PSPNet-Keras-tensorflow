from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D
from keras.layers import merge, concatenate, Lambda, Reshape
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD

from keras.utils import plot_model
import tensorflow as tf

learning_rate = 1e-2 # Could not implement variable learning rate
weight_decay = 5e-4

def BN(name=""):
    return BatchNormalization(momentum=0.95, name=name, epsilon=1e-5)

def Interp(x, size=(60,60)):
    new_height = size[0]
    new_width = size[1]
    resized = tf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized

def Interp_zoom(x, zoom=8):
    old_height = int(x.shape[1])
    old_width = int(x.shape[2])
    new_height = old_height + (old_height-1) * (zoom - 1)
    new_width = old_width + (old_width-1) * (zoom - 1)
    resized = tf.image.resize_images(x, [new_height, new_width], align_corners=True)
    return resized


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv"+lvl+"_"+ sub_lvl +"_1x1_reduce" ,
            "conv"+lvl+"_"+ sub_lvl +"_1x1_reduce_bn",
            "conv"+lvl+"_"+ sub_lvl +"_3x3",
            "conv"+lvl+"_"+ sub_lvl +"_3x3_bn",
            "conv"+lvl+"_"+ sub_lvl +"_1x1_increase",
            "conv"+lvl+"_"+ sub_lvl +"_1x1_increase_bn"]
    if modify_stride == False:
        prev = Conv2D(64 * level, (1,1), strides=(1,1), name=names[0], use_bias=False, kernel_regularizer=l2(weight_decay))(prev)
    elif modify_stride == True:
        prev = Conv2D(64 * level, (1,1), strides=(2,2), name=names[0], use_bias=False, kernel_regularizer=l2(weight_decay))(prev)

    prev = BN(name=names[1])(prev)
    prev = Activation('relu')(prev)

    prev = ZeroPadding2D(padding=(pad,pad))(prev)
    prev = Conv2D(64 * level, (3,3), strides=(1,1), dilation_rate=pad, name=names[2], use_bias=False, kernel_regularizer=l2(weight_decay))(prev)

    prev = BN(name=names[3])(prev)
    prev = Activation('relu')(prev)
    prev = Conv2D(256 * level, (1,1), strides=(1,1), name=names[4], use_bias=False, kernel_regularizer=l2(weight_decay))(prev)
    prev = BN(name=names[5])(prev)
    return prev

def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv"+lvl+"_"+ sub_lvl +"_1x1_proj",
            "conv"+lvl+"_"+ sub_lvl +"_1x1_proj_bn"]

    if modify_stride == False:      
        prev = Conv2D(256 * level ,(1,1), strides=(1,1), name=names[0], use_bias=False, kernel_regularizer=l2(weight_decay))(prev)
    elif modify_stride == True:
        prev = Conv2D(256 * level, (1,1), strides=(2,2), name=names[0], use_bias=False, kernel_regularizer=l2(weight_decay))(prev)

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

    return merge([block_1, block_2], mode='sum') 

def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = Activation('relu')(prev_layer)

    block_1 = residual_conv(prev_layer, level, 
                        pad=pad, lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    return merge([block_1, block_2], mode='sum')

def ResNet(inp):
    #Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
            "conv1_1_3x3_s2_bn",
            "conv1_2_3x3",
            "conv1_2_3x3_bn",
            "conv1_3_3x3",
            "conv1_3_3x3_bn"]

    #---Short branch(only start of network)

    cnv1 = Conv2D(64, (3, 3), strides=(2, 2), padding='same', name=names[0], use_bias=False, kernel_regularizer=l2(weight_decay))(inp) # "conv1_1_3x3_s2"
    bn1 = BN(name=names[1])(cnv1)  # "conv1_1_3x3_s2/bn"
    relu1 = Activation('relu')(bn1)             #"conv1_1_3x3_s2/relu"

    cnv1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', name=names[2], use_bias=False, kernel_regularizer=l2(weight_decay))(relu1) #"conv1_2_3x3"
    bn1 = BN(name=names[3])(cnv1)  #"conv1_2_3x3/bn"
    relu1 = Activation('relu')(bn1)                 #"conv1_2_3x3/relu"

    cnv1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', name=names[4], use_bias=False, kernel_regularizer=l2(weight_decay))(relu1) #"conv1_3_3x3"
    bn1 = BN(name=names[5])(cnv1)      #"conv1_3_3x3/bn"
    relu1 = Activation('relu')(bn1)             #"conv1_3_3x3/relu"

    res = MaxPooling2D(pool_size=(3,3), padding='same', strides=(2,2))(relu1)  #"pool1_3x3_s2"
    
    #---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    #2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1) 
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i+2) 

    #3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True) 
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i+2) 

    #4_1 - 4_6
    res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1) 
    for i in range(5):
        res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i+2) 

    #5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1) 
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i+2)

    res = Activation('relu')(res)
    return res

def interp_block(prev_layer, level, str_lvl=1):

    str_lvl = str(str_lvl)

    names = [
        "conv5_3_pool"+str_lvl+"_conv",
        "conv5_3_pool"+str_lvl+"_conv_bn"
        ]

    kernel = (10*level, 10*level)
    strides = (10*level, 10*level)
    prev_layer = AveragePooling2D(kernel,strides=strides)(prev_layer)
    prev_layer = Conv2D(512, (1,1), strides=(1,1), name=names[0], use_bias=False, kernel_regularizer=l2(weight_decay))(prev_layer)
    prev_layer = BN(name=names[1])(prev_layer)
    prev_layer = Activation('relu')(prev_layer)
    prev_layer = Lambda(Interp)(prev_layer)
    return prev_layer

def PSPNet(res):

    #---PSPNet concat layers with Interpolation

    interp_block1 = interp_block(res, 6, str_lvl=1)
    interp_block2 = interp_block(res, 3, str_lvl=2)
    interp_block3 = interp_block(res, 2, str_lvl=3)
    interp_block6 = interp_block(res, 1, str_lvl=6)

    #concat all these layers by 4th axis(3+1).  resulted shape=(1,60,60,4096)
    res = concatenate([res,
                    interp_block6,
                    interp_block3,
                    interp_block2,
                    interp_block1], axis=3)
    return res

def build_pspnet(activation='softmax'):
    '''
    Normal PSPNet. Consistent up to conv5_4
    '''
    inp = Input((473,473, 3))
    res = ResNet(inp)
    psp = PSPNet(res)

    # Freeze
    features = Model(inputs=inp, outputs=psp)
    for layer in features.layers:
        layer.trainable = False

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv5_4", use_bias=False, kernel_regularizer=l2(weight_decay))(psp)
    x = BN(name="conv5_4_bn")(x)
    x = Activation('relu')(x)
    x = Dropout(0.1)(x)

    x = Conv2D(150, (1, 1), strides=(1, 1), name="conv6", use_bias=False, kernel_regularizer=l2(weight_decay))(x)
    x = Lambda(Interp_zoom)(x)

    loss = None
    if activation == 'softmax':
        x = Activation('softmax')(x)
        loss = 'categorical_crossentropy'
    elif activation == 'sigmoid':
        x = Activation('sigmoid')(x)
        loss = 'binary_crossentropy'

    model = Model(inputs=inp, outputs=x)

    # Solver
    sgd = SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                    loss=loss,
                    metrics=['accuracy'])
    return model
