"""
"""
import tensorflow.keras.backend as K
import tensorflow.keras.layers as KL
from tensorflow_addons.layers import SpectralNormalization as SN

from .layers_se2_conv import SE2Conv_Z2_H, SE2Conv_H_H, OE2Conv_Z2_H_1x1, OE2Conv_H_H_1x1, OE2Conv_Z2_H, OE2Conv_H_H
from .layers_se2_more import SE2Shift, SE2FiLM, CutCorner, OE2Shift, OE2FiLM
from .blocks import CCBN
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import pdb

# -----------------------------------------------------------------------------
# Generator Blocks

def SE2_symmetrize(feat, rot_angles, num_rotations, cut_corner):
    """
    Symmetrizing a probability space using "num_rotations" rotations
    feat: feature of shape [B, H, W, C]
    num_rotations: the number of rotations to symmerize, e.g., 4 or 8
    rot_angles: random integers in [0, num_rotations - 1] of size [B]
    cut_corner: bool; specifying whether the regions outside the disk should be cast to zero -- useful for more than four rotations
    """
    if cut_corner:
        feat = CutCorner()(feat)

    out = tf.zeros_like(feat)
    for k in tf.range(num_rotations):
        idx = (rot_angles == k)
        idx = tf.cast(idx, dtype=tf.float32)
        idx = tf.reshape(idx, [-1,1, 1, 1])
        out = out + KL.RandomRotation([tf.cast(k/num_rotations, dtype=tf.float32), tf.cast(k/num_rotations, dtype=tf.float32)], fill_mode='constant', interpolation='nearest')(feat*idx, training=True)
    return out

def P4_symmetrize(feat, rot_angles):
    """
    Symmetrizing a probability space using 4 rotations
    feat: feature of shape [B, H, W, C]
    rot_angles: random integers in [0, 3] of size [B]
    """

    out = tf.zeros_like(feat)
    for k in tf.range(4):
        idx = (rot_angles == k)
        idx = tf.cast(idx, dtype=tf.float32)
        idx = tf.reshape(idx, [-1,1, 1, 1])
        out = out + tf.image.rot90(feat*idx, k=k)

    return out


def OE2_symmetrize(feat, rot_angles, is_flip, num_rotations, cut_corner):
    """
    E2 Symmetrizing a probability space using "num_rotations" rotations
    feat: feature of shape [B, H, W, C]
    num_rotations: the number of rotations to symmerize, e.g., 4 or 8
    rot_angles: random integers in [0, num_rotations - 1] of size [B]
    is_flip: random integers in [0, 1] of size [B]
    cut_corner: bool; specifying whether the regions outside the disk should be cast to zero -- useful for more than four rotations
    """
    if cut_corner:
        feat = CutCorner()(feat)

    feat_flip = tf.image.flip_left_right(feat)
    
    out = tf.zeros_like(feat)
    
    for k in tf.range(num_rotations):
        idx_0 = tf.math.logical_and((rot_angles == k), (is_flip == 0))
        idx_0 = tf.cast(idx_0, dtype=tf.float32)
        idx_0 = tf.reshape(idx_0, [-1,1, 1, 1])
        out = out + KL.RandomRotation([tf.cast(k/num_rotations, dtype=tf.float32), tf.cast(k/num_rotations, dtype=tf.float32)], fill_mode='constant', interpolation='nearest')(feat*idx_0, training=True)
        
        idx_1 = tf.math.logical_and((rot_angles == k), (is_flip == 1))
        idx_1 = tf.cast(idx_1, dtype=tf.float32)
        idx_1 = tf.reshape(idx_1, [-1,1, 1, 1])
        out = out + KL.RandomRotation([tf.cast(k/num_rotations, dtype=tf.float32), tf.cast(k/num_rotations, dtype=tf.float32)], fill_mode='constant', interpolation='nearest')(feat_flip*idx_1, training=True)
        
    return out


def P4m_symmetrize(feat, rot_angles, is_flip):
    """
    Symmetrizing a probability space using 4 rotations and reflection
    feat: feature of shape [B, H, W, C]
    rot_angles: random integers in [0, 3] of size [B]
    is_flip: random integers in [0, 1] of size [B]
    """

    feat_flip = tf.image.flip_left_right(feat)
    
    out = tf.zeros_like(feat)
    
    for k in tf.range(4):
        idx_0 = tf.math.logical_and((rot_angles == k), (is_flip == 0))
        idx_0 = tf.cast(idx_0, dtype=tf.float32)
        idx_0 = tf.reshape(idx_0, [-1,1, 1, 1])
        out = out + tf.image.rot90(feat*idx_0, k=k)        
        #out = out + KL.RandomRotation([tf.cast(k/num_rotations, dtype=tf.float32), tf.cast(k/num_rotations, dtype=tf.float32)], fill_mode='constant', interpolation='nearest')(feat*idx_0, training=True)
        
        idx_1 = tf.math.logical_and((rot_angles == k), (is_flip == 1))
        idx_1 = tf.cast(idx_1, dtype=tf.float32)
        idx_1 = tf.reshape(idx_1, [-1,1, 1, 1])
        out = out + tf.image.rot90(feat_flip*idx_1, k=k)        
        #out = out + KL.RandomRotation([tf.cast(k/num_rotations, dtype=tf.float32), tf.cast(k/num_rotations, dtype=tf.float32)], fill_mode='constant', interpolation='nearest')(feat_flip*idx_1, training=True)
        
    return out

        

# add optional arg for kernel init
def SE2_CCBN(feat, cla, num_rotations, specnorm=True, initialization='orthogonal'):
    """
    Class-conditional SE2 batch normalization.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        cla: tf Tensor
            Input vector (here latent+condition) to synthesize images with.
        num_rotations: number of rotations
        specnorm: bool
            Whether to use spectral normalization on the linear projections.
        initialization: str
            Kernel initializer for linear projection.
    """
    channels = K.int_shape(feat)[-1] # num of output channels


    if specnorm is True:
        x_beta = SN(KL.Dense(
            channels, kernel_initializer=initialization,
        ))(cla)
        x_gamma = SN(KL.Dense(
            channels, kernel_initializer=initialization,
        ))(cla)
    else:
        x_beta = KL.Dense(channels, kernel_initializer=initialization)(cla)
        x_gamma = KL.Dense(channels, kernel_initializer=initialization)(cla)

    return SE2FiLM(num_rotations)([feat, x_gamma, x_beta])


def OE2_CCBN(feat, cla, num_rotations, specnorm=True, initialization='orthogonal'):
    """
    Class-conditional OE2 batch normalization.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        cla: tf Tensor
            Input vector (here latent+condition) to synthesize images with.
        num_rotations: number of rotations
        specnorm: bool
            Whether to use spectral normalization on the linear projections.
        initialization: str
            Kernel initializer for linear projection.
    """
    channels = K.int_shape(feat)[-1] # num of output channels


    if specnorm is True:
        x_beta = SN(KL.Dense(
            channels, kernel_initializer=initialization,
        ))(cla)
        x_gamma = SN(KL.Dense(
            channels, kernel_initializer=initialization,
        ))(cla)
    else:
        x_beta = KL.Dense(channels, kernel_initializer=initialization)(cla)
        x_gamma = KL.Dense(channels, kernel_initializer=initialization)(cla)

    return OE2FiLM(num_rotations)([feat, x_gamma, x_beta])


def ResBlock_OE2_film(
        feat, cla, nfilters, kernel_size, num_rotations, num_j, num_k, h_input,
        bn_eps=1e-3, upsample=True
):
    """
    A conditionally-modulated upsampling residual block for the Generator.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        cla: tf Tensor
            Input vector (here latent+condition) to synthesize images with.
        nfilters: int
            Number of convolutional filters.
        h_input: str
            Input group. One of {'Z2', 'H'}.
        kernel_size: int
            Convolutional kernel size.
        bn_eps: float
            Numerical constant to keep the BatchNorm denominator happy.
        upsample: bool
            Whether to nearest-neighbors upsample 2x.
    """

    if upsample is True:
        if h_input == 'Z2':
            shortcut = KL.UpSampling2D()(feat)
            shortcut = SN(OE2Conv_Z2_H_1x1(nfilters,
                                           num_rotations))(shortcut)
            shortcut = OE2Shift(num_rotations)(shortcut)
        else:
            # reshape first
            B, H, W, _, num_channels = feat.shape            
            shortcut = KL.Reshape((H, W, num_rotations*2*num_channels))(feat)
            # upsample
            shortcut = KL.UpSampling2D()(shortcut)
            # reshape back
            shortcut = KL.Reshape((2*H, 2*W, num_rotations*2, num_channels))(shortcut)
            shortcut = SN(OE2Conv_H_H_1x1(nfilters,
                                          num_rotations))(shortcut)
            shortcut = OE2Shift(num_rotations)(shortcut)
            
    elif upsample is False:
        if h_input == 'Z2':
            shortcut = SN(OE2Conv_Z2_H_1x1(nfilters,
                                           num_rotations))(feat)
            shortcut = OE2Shift(num_rotations)(shortcut)            
        else:
            shortcut = SN(OE2Conv_H_H_1x1(nfilters,
                                          num_rotations))(feat)
            shortcut = OE2Shift(num_rotations)(shortcut)            
        
    # Convolutional path:
    if h_input == 'Z2':
        skip = KL.BatchNormalization(momentum=0.1, epsilon=bn_eps,
                                     center=False,
                                     scale=False)(feat)
        skip = CCBN(skip, cla, h_input)
    else:
        skip = KL.BatchNormalization(momentum=0.1, epsilon=bn_eps,
                                     center=False,
                                     scale=False)(feat)
        skip = OE2_CCBN(skip, cla, num_rotations)

    skip = KL.Activation('relu')(skip)
    
    if upsample is True:
        if h_input == 'Z2':
            skip = KL.UpSampling2D()(skip)
            #skip = SN(OE2Conv_Z2_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)
        else:
            # reshape first
            B, H, W, _, num_channels = skip.shape
            skip = KL.Reshape((H, W, num_rotations*2*num_channels))(skip)
            # upsample
            skip = KL.UpSampling2D()(skip)
            # reshape back
            skip = KL.Reshape((2*H, 2*W, num_rotations*2, num_channels))(skip)
            #skip = SN(OE2Conv_H_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)
            

    # convolve
    if h_input == 'Z2':
        skip = SN(OE2Conv_Z2_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)
    else:
        skip = SN(OE2Conv_H_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)
            
    skip = KL.BatchNormalization(momentum=0.1)(skip)
    skip = OE2_CCBN(skip, cla, num_rotations)
    skip = KL.Activation('relu')(skip)
    skip = SN(OE2Conv_H_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)

    # Add outputs:
    out = KL.Add()([shortcut, skip])
    
    return out




# -----------------------------------------------------------------------------
# Discriminator blocks

def SE2_discblock(
        feat, nfilters, kernel_size, h_input, num_rotations, num_j, num_k,
        downsample=True, pool='max',
):
    """
    A Conv/SpecNorm/Activation block for the discriminator architectures.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        nfilters: int
            Number of convolutional filters.
        h_input: Z2 or H.
        num_rotations: e.g., 4 or 8.
        num_j and num_k: used to specify the number of basis functions
        kernel_size: int
            Convolutional kernel size.
        downsample: bool
            Whether to downsample 2x.
        pool: str
            Pooling mode. One of {'avg', 'max'}.
    """
    if h_input == 'Z2':
        feat = SN(SE2Conv_Z2_H(nfilters,
                               kernel_size=kernel_size,
                               num_rotations=num_rotations,
                               num_j=num_j,
                               num_k=num_k))(feat)
    elif h_input == 'H':
        feat = SN(SE2Conv_H_H(nfilters,
                               kernel_size=kernel_size,
                               num_rotations=num_rotations,
                               num_j=num_j,
                               num_k=num_k))(feat)
        
    feat = SE2Shift(num_rotations)(feat)
    feat = KL.LeakyReLU(alpha=0.2)(feat)
    if downsample is True:
        B, H, W, _, _ = feat.shape
        feat = KL.Reshape((H, W, num_rotations*nfilters))(feat)
        if pool == 'max':
            feat = KL.MaxPooling2D()(feat)
        elif pool == 'avg':
            feat = KL.AveragePooling2D()(feat)

        # reshape back
        B, H, W, _ = feat.shape
        feat = KL.Reshape((H, W, num_rotations, nfilters))(feat)
    return feat


def ResBlockD_OE2(
        feat, nfilters, kernel_size, num_rotations, num_j, num_k, h_input, BN=False,
        downsample=True, poolshort='max', poolskip='max'
):
    """
    A residual block for the discriminator with spectral and/or batch norm.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        nfilters: int
            Number of convolutional filters.
        h_input: str
            Input group. One of {'Z2', 'C4', 'D4'}.
        h_output: str
            Output group. One of {'Z2', 'C4', 'D4'}.
        pad: str
            Zero-padding mode. One of {'valid', 'same'}.
        stride: int
            Convolutional stride.
        BN: bool
            Whether to use BatchNorm.
        group_equiv: bool
            Whether to be equivariant.
        kernel_size: int
            Convolutional kernel size.
        downsample: bool
            Whether to downsample 2x.
        poolshort: str
            Pooling mode. One of {'avg', 'max'}.
        poolskip: str
            Pooling mode. One of {'avg', 'max'}.
    """

    # Shortcut block:

    if h_input == 'Z2':
        shortcut = SN(OE2Conv_Z2_H_1x1(nfilters,
                                       num_rotations))(feat)
    else:
        shortcut = SN(OE2Conv_H_H_1x1(nfilters,
                                      num_rotations))(feat)

    shortcut = OE2Shift(num_rotations)(shortcut)

    if downsample is True:
        # reshape first
        B, H, W, _, num_channels = shortcut.shape
        shortcut = KL.Reshape((H, W, num_rotations*2*num_channels))(shortcut)
        
        if poolshort == 'max':
            shortcut = KL.MaxPooling2D((2, 2))(shortcut)
        elif poolshort == 'avg':
            shortcut = KL.AveragePooling2D((2, 2))(shortcut)

        # reshape back
        B, H, W, _ = shortcut.shape
        shortcut = KL.Reshape((H, W, num_rotations*2, num_channels))(shortcut)

    # Skip connection:
    if BN is True:
        skip = KL.BatchNormalization(momentum=0.1)(feat)        
        skip = KL.Activation('relu')(skip)
    else:
        skip = KL.Activation('relu')(feat)


    if h_input == 'Z2':
        skip = SN(OE2Conv_Z2_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)
    else:
        skip = SN(OE2Conv_H_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)
        
    if BN is True:
        skip = KL.BatchNormalization(momentum=0.1)(skip)
    else:
        skip = OE2Shift(num_rotations)(skip)

    skip = KL.Activation('relu')(skip)
    skip = SN(OE2Conv_H_H(nfilters, kernel_size=kernel_size, num_rotations=num_rotations, num_j=num_j, num_k=num_k))(skip)
    
    skip = OE2Shift(num_rotations)(skip)   # paper model had no shift here
    
    if downsample is True:
        # reshape first
        B, H, W, _, num_channels = skip.shape
        skip = KL.Reshape((H, W, num_rotations*2*num_channels))(skip)
        
        if poolshort == 'max':
            skip = KL.MaxPooling2D((2, 2))(skip)
        elif poolshort == 'avg':
            skip = KL.AveragePooling2D((2, 2))(skip)

        # reshape back
        B, H, W, _ = skip.shape
        skip = KL.Reshape((H, W, num_rotations*2, num_channels))(skip)


    # Residual addition:
    out = KL.Add()([shortcut, skip])

    
    return out




'''

def ResBlockD(
    feat, nfilters, h_input, h_output, pad='same', stride=1, BN=False,
    group_equiv=True, kernel_size=3, downsample=True,
    poolshort='max', poolskip='max',
):
    """
    A residual block for the discriminator with spectral and/or batch norm.

    Args:
        feat: tf Tensor
            Input feature map to modulate.
        nfilters: int
            Number of convolutional filters.
        h_input: str
            Input group. One of {'Z2', 'C4', 'D4'}.
        h_output: str
            Output group. One of {'Z2', 'C4', 'D4'}.
        pad: str
            Zero-padding mode. One of {'valid', 'same'}.
        stride: int
            Convolutional stride.
        BN: bool
            Whether to use BatchNorm.
        group_equiv: bool
            Whether to be equivariant.
        kernel_size: int
            Convolutional kernel size.
        downsample: bool
            Whether to downsample 2x.
        poolshort: str
            Pooling mode. One of {'avg', 'max'}.
        poolskip: str
            Pooling mode. One of {'avg', 'max'}.
    """
    if group_equiv is True:
        # Shortcut block:
        shortcut = SN(GConv2D(nfilters,
                              kernel_size=1,
                              h_input=h_input,
                              h_output=h_output,
                              strides=stride,
                              padding=pad,
                              kernel_initializer='orthogonal'))(feat)
        shortcut = GShift(h=h_output)(shortcut)
        if downsample is True:
            if poolshort == 'max':
                shortcut = KL.MaxPooling2D((2, 2))(shortcut)
            elif poolshort == 'avg':
                shortcut = KL.AveragePooling2D((2, 2))(shortcut)

        # Skip connection:
        if BN is True:
            if h_input == 'Z2':
                skip = KL.BatchNormalization(momentum=0.1)(feat)
            else:
                skip = GBatchNorm(h=h_input, momentum=0.1)(feat)
            skip = KL.Activation('relu')(skip)
        else:
            skip = KL.Activation('relu')(feat)

        skip = SN(GConv2D(nfilters,
                          kernel_size=kernel_size,
                          h_input=h_input,
                          h_output=h_output,
                          strides=stride,
                          padding=pad,
                          kernel_initializer='orthogonal'))(skip)

        if BN is True:
            skip = GBatchNorm(h=h_output, momentum=0.1)(skip)
        else:
            skip = GShift(h=h_output)(skip)

        skip = KL.Activation('relu')(skip)
        skip = SN(GConv2D(nfilters,
                          kernel_size=kernel_size,
                          h_input=h_output,
                          h_output=h_output,
                          strides=stride,
                          padding=pad,
                          kernel_initializer='orthogonal'))(skip)
        skip = GShift(h=h_output)(skip)   # paper model had no shift here

        if downsample is True:
            if poolskip == 'max':
                skip = KL.MaxPooling2D((2, 2))(skip)
            elif poolskip == 'avg':
                skip = KL.AveragePooling2D((2, 2))(skip)

        # Residual addition:
        out = KL.Add()([shortcut, skip])

    elif group_equiv is False:
        # Shortcut:
        shortcut = SN(KL.Conv2D(nfilters,
                                kernel_size=1,
                                strides=stride,
                                padding=pad,
                                use_bias=True,
                                kernel_initializer='orthogonal'))(feat)
        if downsample is True:
            if poolshort == 'max':
                shortcut = KL.MaxPooling2D((2, 2))(shortcut)
            elif poolshort == 'avg':
                shortcut = KL.AveragePooling2D((2, 2))(shortcut)

        # Skip connection:
        if BN is True:
            skip = KL.BatchNormalization(momentum=0.1)(feat)
            skip = KL.Activation('relu')(skip)
            skip = SN(KL.Conv2D(nfilters,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding=pad,
                                use_bias=False,
                                kernel_initializer='orthogonal'))(skip)
            skip = KL.BatchNormalization(momentum=0.1)(skip)
        else:
            skip = KL.Activation('relu')(feat)
            skip = SN(KL.Conv2D(nfilters,
                                kernel_size=kernel_size,
                                strides=stride,
                                padding=pad,
                                use_bias=True,
                                kernel_initializer='orthogonal'))(skip)

        skip = KL.Activation('relu')(skip)
        skip = SN(KL.Conv2D(nfilters,
                            kernel_size=kernel_size,
                            strides=stride,
                            padding=pad,
                            use_bias=True,  # paper model had this set to False
                            kernel_initializer='orthogonal'))(skip)
        if downsample is True:
            if poolskip == 'max':
                skip = KL.MaxPooling2D((2, 2))(skip)
            elif poolskip == 'avg':
                skip = KL.AveragePooling2D((2, 2))(skip)

        # Residual addition:
        out = KL.Add()([shortcut, skip])

    return out
'''
