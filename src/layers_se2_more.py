from tensorflow.keras import backend as K
import tensorflow.keras.layers as KL
from tensorflow.keras import initializers as initializations
from tensorflow.keras.layers import Layer, InputSpec
from tensorflow.keras.utils import get_custom_objects
from tensorflow.python.keras.layers.pooling import GlobalPooling2D

from groupy.gconv.tensorflow_gconv.splitgconv2d import gconv2d_util
import tensorflow as tf
import numpy as np


# -----------------------------------------------------------------------------
# Conditional scaling and shifting layers:


class SE2FiLM(Layer):
    """
    SE2-equivariant conditional affine transformations for 2D feature maps.
    Modified from:
    stackoverflow.com/questions/55210684/feature-wise-scaling-and-shifting-film-layer-in-keras
    Class design inspired by the implementation of Group Batch Normalization.
    """

    def __init__(self, num_rotations, axis=-1, **kwargs):
        if axis != -1:
            raise ValueError(
                'Assumes 2D input with channels as last dimension.',
            )

        self.num_rotations = num_rotations
        self.axis = axis

        super(SE2FiLM, self).__init__(**kwargs)


    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_gamma_shape, FiLM_beta_shape = input_shape

        self.n_feature_maps = feature_map_shape[-1]

        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]

        assert(int(self.n_feature_maps) == FiLM_gamma_shape[1])
        assert(int(self.n_feature_maps) == FiLM_beta_shape[1])

        super(SE2FiLM, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_gamma, FiLM_beta = x
        # conv_output has shape [B, H, W, theta, lambda]
        # transpose and reshape conv_output to [B, H, W, lambda x theta]
        conv_output_reshaped = tf.transpose(conv_output, [0, 1, 2, 4, 3])
        conv_output_reshaped = tf.reshape(conv_output_reshaped, [-1, self.height, self.width,
                                                                 self.n_feature_maps * self.num_rotations])

        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.tile(FiLM_gamma, [1, self.height, self.width, 1])

        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.tile(FiLM_beta, [1, self.height, self.width, 1])

        def repeat(w):
            return K.reshape(
                K.tile(K.expand_dims(w, -1), [1, 1, 1, 1, self.num_rotations]),
                [-1, self.height, self.width, self.n_feature_maps*self.num_rotations],
            )

        repeated_gamma = repeat(FiLM_gamma)
        repeated_beta = repeat(FiLM_beta)

        # Apply affine transformation
        conv_output_reshaped = (1 + repeated_gamma) * conv_output_reshaped + repeated_beta

        # reshape back
        conv_output = tf.reshape(conv_output_reshaped, [-1, self.height, self.width,
                                                        self.n_feature_maps, self.num_rotations])
        conv_output = tf.transpose(conv_output, [0, 1, 2, 4, 3])

        return conv_output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

    def get_config(self):
        return dict(list({'num_rotations': self.num_rotations}.items()) +
                    list(super(SE2FiLM, self).get_config().items()))



class OE2FiLM(Layer):
    """
    OE2-equivariant conditional affine transformations for 2D feature maps.
    Modified from:
    stackoverflow.com/questions/55210684/feature-wise-scaling-and-shifting-film-layer-in-keras
    Class design inspired by the implementation of Group Batch Normalization.
    """

    def __init__(self, num_rotations, axis=-1, **kwargs):
        if axis != -1:
            raise ValueError(
                'Assumes 2D input with channels as last dimension.',
            )

        self.num_rotations = num_rotations
        self.axis = axis

        super(OE2FiLM, self).__init__(**kwargs)


    def build(self, input_shape):
        assert isinstance(input_shape, list)
        feature_map_shape, FiLM_gamma_shape, FiLM_beta_shape = input_shape

        self.n_feature_maps = feature_map_shape[-1]

        self.height = feature_map_shape[1]
        self.width = feature_map_shape[2]

        assert(int(self.n_feature_maps) == FiLM_gamma_shape[1])
        assert(int(self.n_feature_maps) == FiLM_beta_shape[1])

        super(OE2FiLM, self).build(input_shape)

    def call(self, x):
        assert isinstance(x, list)
        conv_output, FiLM_gamma, FiLM_beta = x
        # conv_output has shape [B, H, W, theta x 2, lambda]
        # transpose and reshape conv_output to [B, H, W, lambda x theta x 2]
        conv_output_reshaped = tf.transpose(conv_output, [0, 1, 2, 4, 3])
        conv_output_reshaped = tf.reshape(conv_output_reshaped, [-1, self.height, self.width,
                                                                 self.n_feature_maps * self.num_rotations * 2])

        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.expand_dims(FiLM_gamma, axis=[1])
        FiLM_gamma = K.tile(FiLM_gamma, [1, self.height, self.width, 1])

        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.expand_dims(FiLM_beta, axis=[1])
        FiLM_beta = K.tile(FiLM_beta, [1, self.height, self.width, 1])

        def repeat(w):
            return K.reshape(
                K.tile(K.expand_dims(w, -1), [1, 1, 1, 1, self.num_rotations * 2]),
                [-1, self.height, self.width, self.n_feature_maps*self.num_rotations*2],
            )

        repeated_gamma = repeat(FiLM_gamma)
        repeated_beta = repeat(FiLM_beta)

        # Apply affine transformation
        conv_output_reshaped = (1 + repeated_gamma) * conv_output_reshaped + repeated_beta

        # reshape back
        conv_output = tf.reshape(conv_output_reshaped, [-1, self.height, self.width,
                                                        self.n_feature_maps, self.num_rotations*2])
        conv_output = tf.transpose(conv_output, [0, 1, 2, 4, 3])

        return conv_output

    def compute_output_shape(self, input_shape):
        assert isinstance(input_shape, list)
        return input_shape[0]

    def get_config(self):
        return dict(list({'num_rotations': self.num_rotations}.items()) +
                    list(super(OE2FiLM, self).get_config().items()))
    


# -----------------------------------------------------------------------------
# Misc. G-equivariant layers covering missing layers in keras-gcnn
# TODO: merge with keras-gcnn

class SE2MaxPool(Layer):
    """Max pool over orientations."""
    def __init__(self, num_rotations, **kwargs):
        super(SE2MaxPool, self).__init__(**kwargs)
        self.num_rotations = num_rotations

    def build(self, input_shape):
        self.shape = input_shape
        super(SE2MaxPool, self).build(input_shape)

    def call(self, x):
        shape = K.int_shape(x)

        max_per_group = K.max(x, -2)

        return max_per_group

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[4],
        )

    def get_config(self):
        config = super(SE2MaxPool, self).get_config()
        config['num_rotations'] = self.num_rotations
        return config

    
class OE2MaxPool(Layer):
    """Max pool over orientations and reflection."""
    def __init__(self, num_rotations, **kwargs):
        super(OE2MaxPool, self).__init__(**kwargs)
        self.num_rotations = num_rotations

    def build(self, input_shape):
        self.shape = input_shape
        super(OE2MaxPool, self).build(input_shape)

    def call(self, x):
        # reflect the second half of x back:
        x_0 = x[..., :self.num_rotations, :]
        x_1_flip = x[..., self.num_rotations:, :]
        B, H, W, _, C = x_1_flip.shape
        x_1_flip_reshape = tf.reshape(x_1_flip, [-1, H, W, self.num_rotations*C])
        x_1_reshape = tf.image.flip_left_right(x_1_flip_reshape)
        x_1 = tf.reshape(x_1_reshape, [-1, H, W, self.num_rotations, C])
        
        x = tf.concat([x_0, x_1], axis=-2)

        max_per_group = K.max(x, -2)

        return max_per_group

    def compute_output_shape(self, input_shape):
        return (
            input_shape[0],
            input_shape[1],
            input_shape[2],
            input_shape[4],
        )

    def get_config(self):
        config = super(OE2MaxPool, self).get_config()
        config['num_rotations'] = self.num_rotations
        return config

    

class SE2Shift(Layer):
    """
    Convolutional bias terms are not currently supported by keras-gcnn, so this
    layer fills that gap, until I update keras-gcnn to support biases.

    Class design inspired by keras-gcnn Group Batch Normalization.
    """

    def __init__(self, num_rotations, axis=-1, beta_initializer='zeros', **kwargs):
        self.num_rotations = num_rotations
        if axis != -1:
            raise ValueError(
                'Assumes 2D input with channels as last dimension.',
            )

        self.axis = axis
        self.beta_initializer = initializations.get(beta_initializer)

        super(SE2Shift, self).__init__(**kwargs)

    def build(self, input_shape):
        # dim is the number of unstructured channels
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                'Axis ' + str(self.axis) + ' of '
                'input tensor should have a defined dimension '
                'but the layer received an input with shape ' +
                str(input_shape) + '.',
            )
        shape = (dim,)

        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer)

        self.broadcast_shape = [1] * len(input_shape)
        self.broadcast_shape[self.axis] = input_shape[self.axis]

        self.built = True

    def call(self, inputs, training=None):

        out = inputs + K.reshape(self.beta, self.broadcast_shape)

        return out

    def get_config(self):
        return dict(list({'num_rotations': self.num_rotations}.items()) +
                    list(super(SE2Shift, self).get_config().items()))


class OE2Shift(Layer):
    """
    Convolutional bias terms are not currently supported by keras-gcnn, so this
    layer fills that gap, until I update keras-gcnn to support biases.

    Class design inspired by keras-gcnn Group Batch Normalization.
    """

    def __init__(self, num_rotations, axis=-1, beta_initializer='zeros', **kwargs):
        self.num_rotations = num_rotations
        if axis != -1:
            raise ValueError(
                'Assumes 2D input with channels as last dimension.',
            )

        self.axis = axis
        self.beta_initializer = initializations.get(beta_initializer)

        super(OE2Shift, self).__init__(**kwargs)

    def build(self, input_shape):
        # dim is the number of unstructured channels
        dim = input_shape[self.axis]
        if dim is None:
            raise ValueError(
                'Axis ' + str(self.axis) + ' of '
                'input tensor should have a defined dimension '
                'but the layer received an input with shape ' +
                str(input_shape) + '.',
            )
        shape = (dim,)

        self.beta = self.add_weight(shape=shape,
                                    name='beta',
                                    initializer=self.beta_initializer)

        self.broadcast_shape = [1] * len(input_shape)
        self.broadcast_shape[self.axis] = input_shape[self.axis]

        self.built = True

    def call(self, inputs, training=None):

        out = inputs + K.reshape(self.beta, self.broadcast_shape)

        return out

    def get_config(self):
        return dict(list({'num_rotations': self.num_rotations}.items()) +
                    list(super(OE2Shift, self).get_config().items()))

    

class CutCorner(Layer):
    """
    Cutting of the corners of an input to keep circular content
    Input is an image of shape [B, H, W, C]
    """

    def __init__(self, **kwargs):
        super(CutCorner, self).__init__(**kwargs)


    def build(self, input_shape):
        H = input_shape[1]
        W = input_shape[2]
        n_channel = input_shape[3]
        if H != W:
            raise ValueError(
                'Assume square input.',
            )

        if len(input_shape) != 4:
            raise ValueError(
                'Assume input of shape [B, H, W, C].',
            )
        if np.mod(H, 2) == 1:
            self.is_odd = True
        else:
            self.is_odd = False
        

    def call(self, inputs):
        B, H, W, C = inputs.shape
        if self.is_odd:
            X, Y = tf.meshgrid(tf.range(-(H//2), H//2+1), tf.range(-(H//2), H//2+1))
            mask = X**2 + Y**2 <= (H//2)**2
        else:
            X, Y = tf.meshgrid(tf.range(-H/2+.5, H/2+.5, 1), tf.range(-H/2+.5, H/2+.5, 1))
            mask = X**2 + Y**2 <= .5**2 + (H/2-.5)**2

            
        mask = tf.cast(mask, tf.float32)
        mask = tf.expand_dims(mask, 0)
        mask = tf.expand_dims(mask, -1)
        return inputs * mask
            
            
        
        
get_custom_objects().update({'SE2FiLM': SE2FiLM})
get_custom_objects().update({'SE2MaxPool': SE2MaxPool})
get_custom_objects().update({'SE2Shift': SE2Shift})
get_custom_objects().update({'OE2FiLM': OE2FiLM})
get_custom_objects().update({'OE2MaxPool': OE2MaxPool})
get_custom_objects().update({'OE2Shift': OE2Shift})
get_custom_objects().update({'CutCorner': CutCorner})
