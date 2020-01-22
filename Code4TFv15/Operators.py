# # Define Operators 
# * **maybe you need to define your own Operators**

import tensorflow as tf
import numpy as np

def tfadd(inputs, options=None):
    return tf.add(inputs[0], inputs[1])


def tfconcat(inputs, options):
    axis = options['axis']
    return tf.concat(inputs, axis)


def tfmaxpool(inputs, options):
    strides = [options['stride_h'], options['stride_w']]
    ksize = [ options['filter_height'],options['filter_width']]
    return tf.nn.max_pool2d(inputs[0],
                            ksize=ksize,
                            strides=strides,
                            padding='SAME')


def tfreshape(inputs, options):
    new_shape = options['new_shape']
    return tf.reshape(inputs[0], shape=new_shape)

def tfdwconv(inputs, options):
    """
    the filters of tf.nn.depthwise_conv2d
        should be `[filter_height, filter_width, in_channels, channel_multiplier]`
    but in tflite it's `[channel_multiplier, filter_height, filter_width, in_channels]` 
    """
    if inputs[1].shape[0] != options['depth_multiplier']:
        print("the depth_multiplier not match")
        raise

    strides = [1, options['stride_h'], options['stride_w'], 1]
    filters = inputs[1].transpose((1, 2, 3, 0))
    filters=filters.astype(np.float32)
    dwconv = tf.nn.depthwise_conv2d(inputs[0],
                                    filter=filters,
                                    strides=strides,
                                    padding='SAME')
    # bias=tf.cast(inputs[2],tf.float32)
    bias=inputs[2].astype(np.float32)
    dwconvbias = tf.nn.bias_add(dwconv, bias)
    return dwconvbias


def tfconv(inputs, options):
    """
    the filters of tf.nn.conv2d_transpose 
        should be `[filter_height, filter_width, in_channels, out_channels]`
    but in tflite it's `[out_channels, filter_height, filter_width, in_channels]` 
    so I transpose it.
    """
    padding = 'SAME'
    if 'padding' in options:
        padding = options['padding']
    strides = [1, options['stride_h'], options['stride_w'], 1]
    filters = inputs[1].transpose((1, 2, 3, 0))
    # filters=tf.cast(filters,tf.float32)
    filters=filters.astype(np.float32)
    conv = tf.nn.conv2d(inputs[0], filter=filters, strides=strides, padding=padding)
    # bias=tf.cast(inputs[2],tf.float32)
    bias=inputs[2].astype(np.float32)
    convbias = tf.nn.bias_add(conv, bias)
    return convbias


def tfrelu(inputs, options=None):
    return tf.nn.relu(inputs[0])


def tfpad(inputs, options=None):
    # pd=tf.to_int32(inputs[1], name='ToInt32')
    # pd=tf.cast(inputs[1],tf.int32)
    pd=inputs[1].astype(np.int32)
    return tf.pad(inputs[0], pd)


def tftransposeconv(inputs, options=None):
    """
    the filters of tf.nn.conv2d_transpose 
        should be `[height, width, output_channels, in_channels]`
    but in tflite it's `[output_channels, height, width, in_channels]`
    """
    insh = inputs[0].shape
    filters = inputs[1].transpose((1, 2, 0, 3))
    # filters=tf.cast(filters,tf.float32)
    filters=filters.astype(np.float32)
    out_shape = [insh[0], insh[1] * 2, insh[2] * 2, filters.shape[2]]
    strides = [1, 2,2 , 1]
    deconv=tf.nn.conv2d_transpose(inputs[0],
                                  filter=filters,
                                  output_shape=out_shape,
                                  strides=strides,
                                  padding='SAME')
    # bias=tf.cast(inputs[2],tf.float32)
    bias=inputs[2].astype(np.float32)
    deconvbias=tf.nn.bias_add(deconv,bias)
    return deconvbias

def tfprelu(inputs, options=None):
    # alphas = tf.cast(inputs[1],tf.float32)
    alphas=inputs[1].astype(np.float32)
#     if len(alphas.shape)!=1:
#         alphas = alphas.reshape(-1)
    x=inputs[0]
    pos = tf.nn.relu(x)
    neg = alphas * (x - tf.abs(x)) * 0.5
    # neg=tf.multiply(alphas,tf.subtract(x,tf.abs(x)))*0.5
    # neg=tf.multiply(alphas,tf.negative(tf.nn.relu(tf.negative(x))))
    
    return tf.add(pos , neg)

def tflogsitic(inputs, options=None):
    return tf.sigmoid(inputs[0])

tf_op_dict = {
    'AddOptions': {
        'op_func': tfadd,
        'in_len': 2
    },
    'ConcatenationOptions': {
        'op_func': tfconcat,
        'in_len': 3
    },
    'Pool2DOptions': {
        'op_func': tfmaxpool,
        'in_len': 1
    },
    'ReshapeOptions': {
        'op_func': tfreshape,
        'in_len': 1
    },
    'DepthwiseConv2DOptions': {
        'op_func': tfdwconv,
        'in_len': 3
    },
    'Conv2DOptions': {
        'op_func': tfconv,
        'in_len': 3
    },
    'RELU': {
        'op_func': tfrelu,
        'in_len': 1
    },
    'PAD': {
        'op_func': tfpad,
        'in_len': 2
    },
    'Convolution2DTransposeBias': {
        'op_func': tftransposeconv,
        'in_len': 3
    },
    'PRELU':{
        'op_func':tfprelu,
        'in_len':2
    },
    'LOGISTIC':{
        'op_func':tflogsitic,
        'in_len':1
    },
}

