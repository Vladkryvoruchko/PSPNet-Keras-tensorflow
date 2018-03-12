from keras import layers
from .tf_layers import adaptive_pooling_2d
from keras.backend import tf as ktf


class Interp(layers.Layer):

    def __init__(self, new_size, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        new_height, new_width = self.new_size
        resized = ktf.image.resize_images(inputs, [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([None, self.new_size[0], self.new_size[1], input_shape[3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        config['new_size'] = self.new_size
        return config


class AdaptivePooling2D(layers.Layer):

    def __init__(self, output_size, mode='avg', **kwargs):
        if mode not in ['avg', 'max']:
            msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
            raise ValueError(msg.format(mode))
        self.output_size = output_size
        self.mode = mode
        super(AdaptivePooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AdaptivePooling2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return adaptive_pooling_2d(inputs, self.output_size, self.mode)

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], self.output_size, self.output_size, input_shape[3]])

    def get_config(self):
        config = super(AdaptivePooling2D, self).get_config()
        config['output_size'] = self.output_size
        config['mode'] = self.mode
        return config
