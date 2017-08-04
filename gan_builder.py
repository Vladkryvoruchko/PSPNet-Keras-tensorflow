from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Flatten
from keras.layers import merge, Lambda, Reshape
from keras.models import Model, Sequential
from keras.regularizers import l2
from keras.optimizers import SGD

from keras.utils import plot_model

import layers_builder as layers
from layers_builder import Interp

class DCGAN:
    def __init__(self):
        self.d_lr = 1e-3
        self.a_lr = 1e-4

        self.img = Input((473,473,3), name="img")
        self.features = layers.PSPNet_features(img)

        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        # self.discriminator = self.build_discriminator_with_features()
        self.adversarial = self.build_adversarial()

    def build_adversarial(self):
        '''
        Discriminator on generator with discriminator frozen
        '''
        pred = self.generator(self.img)

        d_out = self.discriminator(pred)
        # d_out = self.discriminator([pred_scaled, img])
        self.discriminator.trainable = False

        model = Model(inputs=self.img, outputs=[pred,d_out])
        plot_model(model, to_file='adversarial.png')

        # Compile
        opt = SGD(lr=self.a_lr, momentum=0.9, nesterov=True)
        model.compile(optimizer=opt,
                        loss={'pred': 'binary_crossentropy', 'd_out': 'binary_crossentropy'}, 
                        loss_weight={'pred': 1., 'd_out': 1.}
                        metrics=[])

        return model

    def build_generator(self):
        inp = self.features
        x = Conv2D(150, (1, 1), strides=(1, 1), name="conv6")(inp)
        x = Lambda(Interp, arguments={'shape': (473,473)})(x)
        pred = Activation('sigmoid', name="pred")(x)

        model = Model(inputs=self.img, outputs=pred)
        plot_model(model, to_file='gen.png')

        # Only used to generate fake images
        model.compile(optimizer="SGD",loss="binary_crossentropy")
        return model

    def build_discriminator(self):
        '''
        inp:      60x60x150 before bilinear upsampling
        '''
        pred = Input((473,473,150), name="pred")
        pred = Lambda(Interp, arguments={'shape': (60,60)})(pred)

        conv1 = Conv2D(256, 5, strides=1, padding="same", name="d_conv1", activation="relu", use_bias=False)(pred)

        conv1 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv1) # 30x30
        conv2 = Conv2D(512, 3, strides=1, padding="same", name="d_conv2", activation="relu", use_bias=False)(conv1)
        conv2 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv2) # 15x15
        conv3 = Conv2D(1024, 3, strides=1, padding="same", name="d_conv3", activation="relu", use_bias=False)(conv2)
        conv3 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv3) # 8x8
        conv4 = Conv2D(2048, 3, strides=1, padding="same", name="d_conv4", activation="relu", use_bias=False)(conv3)
        conv4 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv4) # 4x4
        conv5 = Conv2D(4096, 3, strides=1, padding="same", name="d_conv5", activation="relu", use_bias=False)(conv4)
        conv5 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv5) # 2x2

        out = Dense(1, activation='sigmoid', name='d_out')(conv5)

        model = Model(inputs=pred, outputs=out)
        plot_model(model, to_file='disc.png')
        return model

    def build_discriminator_with_features(self):
        '''
        pred:      60x60x150 before bilinear upsampling
        features: 60x60x512
        '''
        pred = Input((473,473,150), name="pred")
        pred = Lambda(Interp, arguments={'shape': (60,60)})(pred)

        conv1_1 = Conv2D(256, 5, strides=1, padding="same", name="d_conv1_1", activation="relu", use_bias=False)(pred)
        conv1_2 = Conv2D(256, 5, strides=1, padding="same", name="d_conv1_2", activation="relu", use_bias=False)(self.features)
        conv1 = merge.Concatenate(axis=-1)([conv1_1, conv1_2])

        conv1 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv1) # 30x30
        conv2 = Conv2D(512*2, 3, strides=1, padding="same", name="d_conv2", activation="relu", use_bias=False)(conv1)
        conv2 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv2) # 15x15
        conv3 = Conv2D(1024*2, 3, strides=1, padding="same", name="d_conv3", activation="relu", use_bias=False)(conv2)
        conv3 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv3) # 8x8
        conv4 = Conv2D(2048*2, 3, strides=1, padding="same", name="d_conv4", activation="relu", use_bias=False)(conv3)
        conv4 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv4) # 4x4
        conv5 = Conv2D(4096*2, 3, strides=1, padding="same", name="d_conv5", activation="relu", use_bias=False)(conv4)
        conv5 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv5) # 2x2

        out = Dense(1, activation='sigmoid', name='d_out')(conv5)

        model = Model(inputs=[pred,self.img], outputs=out)
        plot_model(model, to_file='disc_with_features.png')
        return model



