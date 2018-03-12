import tensorflow as tf


def adaptive_pooling_2d(inputs, output_size: int, mode: str):
    """
    Performs a pooling operation that results in a fixed size:
    output_size x output_size.

    Used by spatial_pyramid_pool. Refer to appendix A in [1].

    Args:
        inputs: A 4D Tensor (B, H, W, C)
        output_size: The output size of the pooling operation.
        mode: The pooling mode {max, avg}

    Returns:
        A list of tensors, for each output bin.
        The list contains output_size * output_size elements, where
        each elment is a Tensor (N, C).

    References:
        [1] He, Kaiming et al (2015):
            Spatial Pyramid Pooling in Deep Convolutional Networks
            for Visual Recognition.
            https://arxiv.org/pdf/1406.4729.pdf.

    Ported from: https://github.com/luizgh/Lasagne/commit/c01e3d922a5712ca4c54617a15a794c23746ac8c
    """
    inputs_shape = tf.shape(inputs)
    batch = tf.cast(tf.gather(inputs_shape, 0), tf.int32)
    h = tf.cast(tf.gather(inputs_shape, 1), tf.int32)
    w = tf.cast(tf.gather(inputs_shape, 2), tf.int32)
    channels = tf.cast(tf.gather(inputs_shape, 3), tf.int32)
    if mode == 'max':
        pooling_op = tf.reduce_max
    elif mode == 'avg':
        pooling_op = tf.reduce_mean
    else:
        msg = "Mode must be either 'max' or 'avg'. Got '{0}'"
        raise ValueError(msg.format(mode))
    result = []
    n = output_size
    for row in range(output_size):
        for col in range(output_size):
            # start_h = floor(row / n * h)
            start_h = tf.cast(
                tf.floor(tf.multiply(tf.divide(row, n), tf.cast(h, tf.float32))), tf.int32)
            # end_h = ceil((row + 1) / n * h)
            end_h = tf.cast(
                tf.ceil(tf.multiply(tf.divide((row + 1), n), tf.cast(h, tf.float32))), tf.int32)
            # start_w = floor(col / n * w)
            start_w = tf.cast(
                tf.floor(tf.multiply(tf.divide(col, n), tf.cast(w, tf.float32))), tf.int32)
            # end_w = ceil((col + 1) / n * w)
            end_w = tf.cast(
                tf.ceil(tf.multiply(tf.divide((col + 1), n), tf.cast(w, tf.float32))), tf.int32)
            pooling_region = inputs[:, start_h:end_h, start_w:end_w, :]
            pool_result = pooling_op(
                pooling_region, axis=(1, 2), keepdims=True)
            result.append(pool_result)
    return tf.reshape(tf.concat(result, axis=1), [batch, output_size, output_size, channels])
