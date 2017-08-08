from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Dense
from keras.layers import BatchNormalization, Activation, Input, Dropout, ZeroPadding2D, Flatten
from keras.layers import Lambda, Reshape
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD

import layers_builder as layers
from layers_builder import Interp

class DCGAN:
    def __init__(self, disc_use_features=False):
        self.d_lr = 1e-3
        self.a_lr = 1e-2
        self.disc_use_features = disc_use_features

        self.img = Input((473,473,3), name="img")
        self.features = layers.PSPNet_features(self.img)

        self.generator = self.build_generator()
        if self.disc_use_features:
            self.discriminator = self.build_discriminator_with_features()
        else:
            self.discriminator = self.build_discriminator()
        self.adversarial = self.build_adversarial()

    def build_adversarial(self):
        '''
        Discriminator on generator with discriminator frozen
        '''
        pred = self.generator(self.img)
        pred = Activation('linear', name="pred")(pred)

        d_out = None
        if self.disc_use_features:
            d_out = self.discriminator([pred, self.img])
        else:
            d_out = self.discriminator(pred)
        d_out = Activation('linear', name="d_out")(d_out)
        self.discriminator.trainable = False

        model = Model(inputs=self.img, outputs=[pred,d_out])
        #model = Model(inputs=self.img, outputs=d_out)

        # Compile
        opt = SGD(lr=self.a_lr, momentum=0, nesterov=True)
        #model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer=opt,
                        loss={'pred': 'binary_crossentropy', 'd_out': 'binary_crossentropy'}, 
                        loss_weights={'pred': 10., 'd_out': 1.},
                        metrics=['accuracy'])
        return model

    def build_generator(self):
        inp = self.features
        x = Conv2D(150, (1, 1), strides=(1, 1), name="conv6")(inp)
        x = Lambda(Interp, arguments={'shape': (473,473)})(x)
        pred = Activation('sigmoid')(x)

        model = Model(inputs=self.img, outputs=pred)

        # Only used to generate fake images
        model.compile(optimizer="SGD", loss="binary_crossentropy", metrics=['accuracy'])
        return model

    def build_discriminator(self):
        '''
        inp:      60x60x150 before bilinear upsampling
        '''
        pred = Input((473,473,150), name="pred")
        pred_scaled = Lambda(Interp, arguments={'shape': (60,60)})(pred)

        conv1 = Conv2D(256, 5, strides=1, padding="same", name="d_conv1", activation="relu", use_bias=False)(pred_scaled)
        conv1 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv1) # 30x30
        conv2 = Conv2D(256, 3, strides=1, padding="same", name="d_conv2", activation="relu", use_bias=False)(conv1)
        conv2 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv2) # 15x15
        conv3 = Conv2D(512, 3, strides=1, padding="same", name="d_conv3", activation="relu", use_bias=False)(conv2)
        conv3 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv3) # 8x8
        conv4 = Conv2D(512, 3, strides=1, padding="same", name="d_conv4", activation="relu", use_bias=False)(conv3)
        conv4 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv4) # 4x4
        conv5 = Conv2D(1024, 3, strides=1, padding="same", name="d_conv5", activation="relu", use_bias=False)(conv4)
        conv5 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv5) # 2x2

        flatten = Flatten()(conv5)
        out = Dense(1, activation='sigmoid', name='d_out')(flatten)

        model = Model(inputs=pred, outputs=out)
        opt = SGD(lr=self.d_lr, momentum=0, nesterov=True)
        model.compile(optimizer=opt,
                        loss='binary_crossentropy', 
                        metrics=['accuracy'])
        return model

    def build_discriminator_with_features(self):
        '''
        pred:      60x60x150 before bilinear upsampling
        features: 60x60x512
        '''
        pred = Input((473,473,150), name="pred")
        pred_scaled = Lambda(Interp, arguments={'shape': (60,60)})(pred)

        # Merge with features
        conv1_1 = Conv2D(256, 5, strides=1, padding="same", name="d_conv1_1", activation="relu", use_bias=False)(pred_scaled)
        conv1_2 = Conv2D(256, 5, strides=1, padding="same", name="d_conv1_2", activation="relu", use_bias=False)(self.features)
        conv1 = Concatenate(axis=-1)([conv1_1, conv1_2])

        conv1 = Conv2D(256*2, 5, strides=1, padding="same", name="d_conv1", activation="relu", use_bias=False)(conv1)
        conv1 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv1) # 30x30
        conv2 = Conv2D(256*2, 3, strides=1, padding="same", name="d_conv2", activation="relu", use_bias=False)(conv1)
        conv2 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv2) # 15x15
        conv3 = Conv2D(512*2, 3, strides=1, padding="same", name="d_conv3", activation="relu", use_bias=False)(conv2)
        conv3 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv3) # 8x8
        conv4 = Conv2D(512*2, 3, strides=1, padding="same", name="d_conv4", activation="relu", use_bias=False)(conv3)
        conv4 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv4) # 4x4
        conv5 = Conv2D(1024*2, 3, strides=1, padding="same", name="d_conv5", activation="relu", use_bias=False)(conv4)
        conv5 = MaxPooling2D(pool_size=2, padding='same', strides=2)(conv5) # 2x2

        flatten = Flatten()(conv5)
        out = Dense(1, activation='sigmoid', name='d_out')(flatten)

        model = Model(inputs=[pred,self.img], outputs=out)
        opt = SGD(lr=self.d_lr, momentum=0, nesterov=True)
        model.compile(optimizer=opt,
                        loss='binary_crossentropy',
                        metrics=['accuracy'])
        return model



