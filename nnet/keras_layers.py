from keras import layers
from keras.backend import tf as ktf
from .tf_layers import adaptive_pooling_2d


class Interp(layers.Layer):
    """Bilinear interpolation
    __call__ Takes two params. First param is layer we need to resize.
    Second param is tensor which shape is target.
    """

    def __init__(self, new_size=None, **kwargs):
        self.new_size = new_size
        super(Interp, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Interp, self).build(input_shape)

    def call(self, inputs, **kwargs):
        assert(len(inputs) == 2)
        shape = ktf.shape(inputs[1])
        new_height, new_width = shape[1], shape[2]
        resized = ktf.image.resize_images(inputs[0], [new_height, new_width],
                                          align_corners=True)
        return resized

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0][0], None, None, input_shape[0][3]])

    def get_config(self):
        config = super(Interp, self).get_config()
        return config


class AdaptivePooling2D(layers.Layer):

    def __init__(self, out_size, mode='avg', **kwargs):
        if mode not in ['avg', 'max']:
            msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
            raise ValueError(msg.format(mode))
        self.out_size = out_size
        self.mode = mode
        super(AdaptivePooling2D, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AdaptivePooling2D, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return adaptive_pooling_2d(inputs, self.out_size, self.mode)

    def compute_output_shape(self, input_shape):
        return tuple([input_shape[0], self.out_size, self.out_size, input_shape[3]])

    def get_config(self):
        config = super(AdaptivePooling2D, self).get_config()
        config['out_size'] = self.out_size
        config['mode'] = self.mode
        return config
