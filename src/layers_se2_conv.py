import tensorflow as tf
from tensorflow import keras
from keras import regularizers
import numpy as np
import os
from tensorflow_addons.layers import SpectralNormalization as SN
from tensorflow.keras import backend as K
#from layers_se2_more import SE2MaxPool, OE2MaxPool

import pdb
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


tf.config.run_functions_eagerly(False)


def cartesian_to_polar_coordinates(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return (phi, rho)

def rotated_basis_funcs(x, y, rot, j, k, real_or_imag='real', sigma=.5):
    """
    x, y are 1d arrays (of length N) of cartesian coordiates
    returns  real(psi_{j,k}(R_(-theta)(x,y))) or imagin(psi_{j,k}(R_(-theta)(x,y))), depending on the type of real_or_imag
    """
    theta, rho = cartesian_to_polar_coordinates(x, y)
    func = np.exp(-(rho-j)**2/(2*sigma**2))
    if k == 0:
        return func
    else:
        if real_or_imag == 'real':
            func *= np.cos(k*(theta - rot))
            return func
        elif real_or_imag == 'imag':
            func *= np.sin(k*(theta - rot))
            return func
        else:
            print('wrong!')

def steerable_basis_fixed_rot(kernel_size, rot, num_j, num_k):
    """
    Return the basis funcs at a fixed rotation
    Array is of shape [ks, ks, num_funcs = 1 + (num_j-1) * (2 num_k - 1)]
    """
    X, Y = np.meshgrid(range(-(kernel_size // 2), (kernel_size // 2)+1), range(-(kernel_size // 2), (kernel_size // 2)+1))
    ugrid = np.concatenate([Y.reshape(-1,1), X.reshape(-1,1)], 1)

    bxy = []
    # adding the first basis functions with no angular frequency, i.e., k=0
    for j in range(num_j):
        bxy.append(rotated_basis_funcs(ugrid[:,0], ugrid[:,1], rot, j, 0))

    # adding the remaining basis functions with non-zero angular frequency, i.e., k > 0
    for j in range(1, num_j):
        for k in range(1, num_k):
            bxy.append(rotated_basis_funcs(ugrid[:,0], ugrid[:,1], rot, j, k, 'real'))
            bxy.append(rotated_basis_funcs(ugrid[:,0], ugrid[:,1], rot, j, k, 'imag'))

    basis = np.array(bxy).reshape(-1, kernel_size, kernel_size)
    basis = basis.transpose((1,2,0))    

    return basis
    

def steerable_basis(kernel_size, num_rotations, num_j, num_k):
    """ 
    basis is in shape: [ks, ks, theta, num_funs = 1 + (num_j -1) * (2 num_k - 1)]
    """
    rotations =  2* np.pi /num_rotations *np.arange(num_rotations)
    basis_tensors = []
    for rot in rotations:
        basis = steerable_basis_fixed_rot(kernel_size, rot, num_j, num_k)
        basis_tensors.append(basis)

    # steerable_basis has shape [num_rot, ks, ks, num_funcs]
    steerable_basis = tf.convert_to_tensor(basis_tensors, dtype=tf.float32)
    
    # transpose to [ks, ks, theta, num_funs]
    steerable_basis = tf.transpose(steerable_basis, [1, 2, 0, 3])
    

    # normalize
    norm = ((tf.math.reduce_sum(steerable_basis**2, [0, 1], keepdims=True))**.5)[:, :, 0:1, :]
    return steerable_basis/norm

# debugging
#num_rots = 8
#def show_images(imgs):
#    for i in range(num_rots):
#        for j in range(15):
#            idx = (i)*15 + j +1
#            plt.subplot(num_rots, 15, idx)
#            plt.imshow(imgs[:,:,i,j].numpy())
#            plt.axis('off')
#            plt.colorbar()
#    plt.savefig('haha.png')
    
#tmp = steerable_basis(7, num_rots, 3, 4)
#show_images(tmp)
#pdb.set_trace()    
    

class SE2Conv_Z2_H(keras.layers.Layer):

    def __init__(self, filters, kernel_size, num_rotations, num_j, num_k, strides=1, padding='same', trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(SE2Conv_Z2_H, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            kernel_size: An odd integer, typically 5, i.e., 5x5 spatial kernels
            strides: typically 1
            padding: typically 'same'

        """
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations    
        self.strides = strides
        if padding == 'same' or 'SAME':
           self.padding = 'SAME'
        else:
            raise NotImplementedError
        self.basis = steerable_basis(kernel_size, num_rotations, num_j, num_k)
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])
        
        num_funs = self.basis.shape[-1]
        # weight is called kernel, as we need to use spectral normalization
        # kernel is in shape [num_funs, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[num_funs, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.kernel_size**2/self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True

    def call(self, inputs):
        basis = self.basis
        weight = self.kernel
        
        # the (convolutional) kernel is of the shape [ks, ks, num_rotations, in, out]
        kernel = tf.tensordot(basis, weight, 1)
        # permute the kernel to [ks, ks, in, num_rotations x out]
        kernel = tf.transpose(kernel, perm=[0, 1, 3, 2, 4])
        kernel = tf.reshape(kernel, [self.kernel_size, self.kernel_size, -1, self.num_rotations * self.filters])

        # convolution
        outputs = tf.nn.convolution(inputs, kernel, strides=self.strides, padding=self.padding)
        B, H, W, _ = outputs.shape

        # outputs in shape [B, H, W, num_rotations, out]
        outputs = tf.reshape(outputs, [-1, H, W, self.num_rotations, self.filters])
        #outputs = tf.reshape(outputs, [-1, H, W, self.num_rotations, self.filters])

        return outputs

## debugging:
#num_rotations = 8
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train = np.expand_dims(x_train, axis=3)

#model = SE2Conv_Z2_H(filters=2, kernel_size=5, num_rotations=num_rotations, num_j=3, num_k=4, strides=1)
#x_train = tf.convert_to_tensor(x_train/255., dtype=tf.float32)
#x_train_rot = tf.image.rot90(x_train)


#y = tf.reshape(model(x_train), [60000, 28, 28, -1])
#y_rot = tf.reshape(model(x_train_rot), [60000, 28, 28, -1])
#y_rot = tf.image.rot90(y_rot, 3)

#def show_images(y1, y2, idx):
#    for i in range(num_rotations):
#        plt.subplot(num_rotations, 2, 2*i+1)
#        plt.imshow(y1[idx, :, :, 2*i], cmap='jet')
#        plt.colorbar()        
#        plt.subplot(num_rotations, 2, 2*i+2)
#        plt.imshow(y2[idx, :, :, 2*i], cmap='jet')
#        plt.colorbar()
#    plt.savefig('haha.png')

#    show_images(y, y_rot, 2)    
#pdb.set_trace()


class OE2Conv_Z2_H(keras.layers.Layer):

    def __init__(self, filters, kernel_size, num_rotations, num_j, num_k, strides=1, padding='same', trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(OE2Conv_Z2_H, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            kernel_size: An odd integer, typically 5, i.e., 5x5 spatial kernels
            strides: typically 1
            padding: typically 'same'

        """
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations    
        self.strides = strides
        if padding == 'same' or 'SAME':
           self.padding = 'SAME'
        else:
            raise NotImplementedError
        self.basis = steerable_basis(kernel_size, num_rotations, num_j, num_k)
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])
        
        num_funs = self.basis.shape[-1]
        # weight is called kernel, as we need to use spectral normalization
        # kernel is in shape [num_funs, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[num_funs, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.kernel_size**2/self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True

    def call(self, inputs):
        basis = self.basis
        weight = self.kernel
        
        # the (convolutional) kernel is of the shape [ks, ks, num_rotations, in, out]
        kernel = tf.tensordot(basis, weight, 1)
        # permute the kernel to [ks, ks, in, num_rotations x out]
        kernel = tf.transpose(kernel, perm=[0, 1, 3, 2, 4])
        kernel = tf.reshape(kernel, [self.kernel_size, self.kernel_size, -1, self.num_rotations * self.filters])

        # inputs_flip
        inputs_flip = tf.image.flip_left_right(inputs)

        # convolution
        outputs = tf.nn.convolution(inputs, kernel, strides=self.strides, padding=self.padding)
        B, H, W, _ = outputs.shape

        outputs_flip = tf.nn.convolution(inputs_flip, kernel, strides=self.strides, padding=self.padding)

        # outputs in shape [B, H, W, num_rotations x 2, out]
        outputs = tf.reshape(outputs, [-1, H, W, self.num_rotations, self.filters])
        outputs_flip = tf.reshape(outputs_flip, [-1, H, W, self.num_rotations, self.filters])

        # outputs = combined together
        outputs = tf.concat([outputs, outputs_flip], axis=-2)
        
        return outputs


class SE2Conv_Z2_H_1x1(keras.layers.Layer):

    def __init__(self, filters, num_rotations, strides=1, trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(SE2Conv_Z2_H_1x1, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            strides: typically 1
        """
        
        self.filters = filters
        self.num_rotations = num_rotations    
        self.strides = strides
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])

        # kernel is in shape [1, 1, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[1, 1, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True

    def call(self, inputs):
        # kernel is in shape [1, 1, in, out]
        kernel = self.kernel

        # convolution
        outputs = tf.nn.convolution(inputs, kernel, strides=self.strides, padding='SAME')

        outputs = tf.expand_dims(outputs, axis=-2)
        multiples = tf.constant([1, 1, 1, self.num_rotations, 1])

        # outputs in shape [B, H, W, num_rotations, out]
        outputs = tf.tile(outputs, multiples)

        return outputs


class OE2Conv_Z2_H_1x1(keras.layers.Layer):

    def __init__(self, filters, num_rotations, strides=1, trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(OE2Conv_Z2_H_1x1, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            strides: typically 1
        """
        
        self.filters = filters
        self.num_rotations = num_rotations    
        self.strides = strides
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])

        # kernel is in shape [1, 1, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[1, 1, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True

    def call(self, inputs):
        # kernel is in shape [1, 1, in, out]
        kernel = self.kernel

        # inputs_flip
        inputs_flip = tf.image.flip_left_right(inputs)        

        # convolution
        outputs = tf.nn.convolution(inputs, kernel, strides=self.strides, padding='SAME')
        outputs_flip = tf.nn.convolution(inputs_flip, kernel, strides=self.strides, padding='SAME')

        outputs = tf.expand_dims(outputs, axis=-2)
        outputs_flip = tf.expand_dims(outputs_flip, axis=-2)        
        multiples = tf.constant([1, 1, 1, self.num_rotations, 1])

        # outputs in shape [B, H, W, num_rotations, out]
        outputs = tf.tile(outputs, multiples)
        outputs_flip = tf.tile(outputs_flip, multiples)

        # outputs = combined together
        outputs = tf.concat([outputs, outputs_flip], axis=-2)        

        return outputs
    


class SE2Conv_H_H(keras.layers.Layer):

    def __init__(self, filters, kernel_size, num_rotations, num_j, num_k, strides=1, padding='same', trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(SE2Conv_H_H, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            kernel_size: An odd integer, typically 5, i.e., 5x5 spatial kernels
            strides: typically 1
            padding: typically 'same'

        """
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations    
        self.strides = strides
        if padding == 'same' or 'SAME':
           self.padding = 'SAME'
        else:
            raise NotImplementedError
        self.basis = steerable_basis(kernel_size, num_rotations, num_j, num_k)
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])
        
        num_funs = self.basis.shape[-1]
        # weight is called kernel, as we need to use spectral normalization
        # kernel is in shape [num_funs, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[num_funs, self.num_rotations, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.kernel_size**2/self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True


    def call(self, inputs):
        basis = self.basis
        weight = self.kernel

        # the (convolutional) kernel is of the shape [ks, ks, num_rotations, num_rotations, in, out]
        # indexed by [u1, u2, theta, theta', lambda', lambda]
        kernel = tf.tensordot(basis, weight, 1)
        # permute the kernel to [ks, ks, num_rotations, in, num_rotations x out]
        # indexed by [u1, u2, theta', lambda', theta x lambda]
        kernel = tf.transpose(kernel, perm=[0, 1, 3, 4, 2, 5])
        kernel = tf.reshape(kernel, [self.kernel_size, self.kernel_size, self.num_rotations, -1, self.num_rotations * self.filters])

        # calculate circular padding of the inputs in the rotation channel
        # inputs of shape [B, H, W, num_rotations x in]
        B, H, W, rot, input_channel = inputs.shape
        inputs = tf.concat([inputs, inputs], axis=-2)
        
        # convolution

        # outputs = tf.zeros([B, H, W, self.num_rotations*self.filters], dtype=tf.float32)
        inputs_ = inputs[:, :, :, 0: self.num_rotations, :]
        inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
        kernel_ = kernel[:, :, 0, :, :]
        outputs = tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding=self.padding)
        
        for i in tf.range(1, self.num_rotations):
            inputs_ = inputs[:, :, :, i:i + self.num_rotations, :]
            inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
            kernel_ = kernel[:, :, i, :, :]
            outputs += tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding=self.padding)        

        outputs = tf.reshape(outputs, [-1, H, W, self.num_rotations, self.filters])
        return outputs


class OE2Conv_H_H(keras.layers.Layer):

    def __init__(self, filters, kernel_size, num_rotations, num_j, num_k, strides=1, padding='same', trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(OE2Conv_H_H, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            kernel_size: An odd integer, typically 5, i.e., 5x5 spatial kernels
            strides: typically 1
            padding: typically 'same'

        """
        
        self.filters = filters
        self.kernel_size = kernel_size
        self.num_rotations = num_rotations    
        self.strides = strides
        if padding == 'same' or 'SAME':
           self.padding = 'SAME'
        else:
            raise NotImplementedError
        self.basis = steerable_basis(kernel_size, num_rotations, num_j, num_k)
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])
        
        num_funs = self.basis.shape[-1]
        # weight is called kernel, as we need to use spectral normalization
        # kernel is in shape [num_funs, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[num_funs, self.num_rotations, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.kernel_size**2/self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True


    def call(self, inputs):
        basis = self.basis
        weight = self.kernel

        # the (convolutional) kernel is of the shape [ks, ks, num_rotations, num_rotations, in, out]
        # indexed by [u1, u2, theta, theta', lambda', lambda]
        kernel = tf.tensordot(basis, weight, 1)
        # permute the kernel to [ks, ks, num_rotations, in, num_rotations x out]
        # indexed by [u1, u2, theta', lambda', theta x lambda]
        kernel = tf.transpose(kernel, perm=[0, 1, 3, 4, 2, 5])
        kernel = tf.reshape(kernel, [self.kernel_size, self.kernel_size, self.num_rotations, -1, self.num_rotations * self.filters])

        # split input into 2
        inputs_flip = inputs[..., self.num_rotations:, :]
        inputs = inputs[..., :self.num_rotations, :]        

        # calculate circular padding of the inputs in the rotation channel
        B, H, W, rot, input_channel = inputs.shape
        inputs = tf.concat([inputs, inputs], axis=-2)
        inputs_flip = tf.concat([inputs_flip, inputs_flip], axis=-2)
        
        # convolution

        # outputs = tf.zeros([B, H, W, self.num_rotations*self.filters], dtype=tf.float32)
        inputs_ = inputs[:, :, :, 0: self.num_rotations, :]
        inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
        kernel_ = kernel[:, :, 0, :, :]
        outputs = tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding=self.padding)
        
        for i in tf.range(1, self.num_rotations):
            inputs_ = inputs[:, :, :, i:i + self.num_rotations, :]
            inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
            kernel_ = kernel[:, :, i, :, :]
            outputs += tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding=self.padding)        

        outputs = tf.reshape(outputs, [-1, H, W, self.num_rotations, self.filters])

        # convolution for flip
        inputs_flip_ = inputs_flip[:, :, :, 0: self.num_rotations, :]
        inputs_flip_ = tf.reshape(inputs_flip_, [-1, H, W, self.num_rotations * input_channel])
        kernel_ = kernel[:, :, 0, :, :]
        outputs_flip = tf.nn.convolution(inputs_flip_, kernel_, strides=self.strides, padding=self.padding)
        
        for i in tf.range(1, self.num_rotations):
            inputs_flip_ = inputs_flip[:, :, :, i:i + self.num_rotations, :]
            inputs_flip_ = tf.reshape(inputs_flip_, [-1, H, W, self.num_rotations * input_channel])
            kernel_ = kernel[:, :, i, :, :]
            outputs_flip += tf.nn.convolution(inputs_flip_, kernel_, strides=self.strides, padding=self.padding)        

        outputs_flip = tf.reshape(outputs_flip, [-1, H, W, self.num_rotations, self.filters])

        # outputs = combined together
        outputs = tf.concat([outputs, outputs_flip], axis=-2)
        
        return outputs



class SE2Conv_H_H_1x1(keras.layers.Layer):

    def __init__(self, filters, num_rotations, strides=1, trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(SE2Conv_H_H_1x1, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            strides: typically 1
        """
        
        self.filters = filters
        self.num_rotations = num_rotations    
        self.strides = strides
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])

        # kernel is in shape [1, 1, num_rotations, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[1, 1, self.num_rotations, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True

    def call(self, inputs):
        # kernel is in shape [1, 1, num_rotations, in, out]
        kernel = self.kernel

        # tile kernel to shape [1, 1, num_rotations, in, theta x out]
        multiples = tf.constant([1, 1, 1, 1, self.num_rotations])
        kernel = tf.tile(kernel, multiples)

        # calculate circular padding of the inputs in the rotation channel
        B, H, W, rot, input_channel = inputs.shape
        inputs = tf.concat([inputs, inputs], axis=-2)

        # outputs = tf.zeros([B, H, W, self.num_rotations*self.filters], dtype=tf.float32)
        inputs_ = inputs[:, :, :, 0: self.num_rotations, :]
        inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
        kernel_ = kernel[:, :, 0, :, :]
        outputs = tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding='SAME')
        
        for i in tf.range(1, self.num_rotations):
            inputs_ = inputs[:, :, :, i:i + self.num_rotations, :]
            inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
            kernel_ = kernel[:, :, i, :, :]
            outputs += tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding='SAME')        

        outputs = tf.reshape(outputs, [-1, H, W, self.num_rotations, self.filters])
        return outputs



class OE2Conv_H_H_1x1(keras.layers.Layer):

    def __init__(self, filters, num_rotations, strides=1, trainable=True,
                 name=None, activity_regularizer=None, **kwargs):
        super(OE2Conv_H_H_1x1, self).__init__(
            trainable=trainable, name=name, activity_regularizer=regularizers.get(activity_regularizer),
            **kwargs)
        """
        Args:
            filters: An integer, the number of unstructured channels
            strides: typically 1
        """
        
        self.filters = filters
        self.num_rotations = num_rotations    
        self.strides = strides
        
    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_channel = int(input_shape[-1])

        # kernel is in shape [1, 1, num_rotations, in, out]        
        self.kernel = self.add_weight(
            name='weight',
            shape=[1, 1, self.num_rotations, input_channel, self.filters],
            initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=np.sqrt(2./self.num_rotations/input_channel)),
            trainable=True,
            dtype=tf.float32)
        
        self.built=True

    def call(self, inputs):
        # kernel is in shape [1, 1, num_rotations, in, out]
        kernel = self.kernel

        # tile kernel to shape [1, 1, num_rotations, in, theta x out]
        multiples = tf.constant([1, 1, 1, 1, self.num_rotations])
        kernel = tf.tile(kernel, multiples)

        # split input into 2
        inputs_flip = inputs[..., self.num_rotations:, :]
        inputs = inputs[..., :self.num_rotations, :]        

        # calculate circular padding of the inputs in the rotation channel
        B, H, W, rot, input_channel = inputs.shape
        inputs = tf.concat([inputs, inputs], axis=-2)
        inputs_flip = tf.concat([inputs_flip, inputs_flip], axis=-2)

        # convolution

        # outputs = tf.zeros([B, H, W, self.num_rotations*self.filters], dtype=tf.float32)
        inputs_ = inputs[:, :, :, 0: self.num_rotations, :]
        inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
        kernel_ = kernel[:, :, 0, :, :]
        outputs = tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding='SAME')
        
        for i in tf.range(1, self.num_rotations):
            inputs_ = inputs[:, :, :, i:i + self.num_rotations, :]
            inputs_ = tf.reshape(inputs_, [-1, H, W, self.num_rotations * input_channel])
            kernel_ = kernel[:, :, i, :, :]
            outputs += tf.nn.convolution(inputs_, kernel_, strides=self.strides, padding='SAME')        

        outputs = tf.reshape(outputs, [-1, H, W, self.num_rotations, self.filters])

        # convolution for flip
        inputs_flip_ = inputs_flip[:, :, :, 0: self.num_rotations, :]
        inputs_flip_ = tf.reshape(inputs_flip_, [-1, H, W, self.num_rotations * input_channel])
        kernel_ = kernel[:, :, 0, :, :]
        outputs_flip = tf.nn.convolution(inputs_flip_, kernel_, strides=self.strides, padding='SAME')
        
        for i in tf.range(1, self.num_rotations):
            inputs_flip_ = inputs_flip[:, :, :, i:i + self.num_rotations, :]
            inputs_flip_ = tf.reshape(inputs_flip_, [-1, H, W, self.num_rotations * input_channel])
            kernel_ = kernel[:, :, i, :, :]
            outputs_flip += tf.nn.convolution(inputs_flip_, kernel_, strides=self.strides, padding='SAME')        

        outputs_flip = tf.reshape(outputs_flip, [-1, H, W, self.num_rotations, self.filters])

        # outputs = combined together
        outputs = tf.concat([outputs, outputs_flip], axis=-2)
        
        return outputs    

    

    
## debugging:
#num_rotations = 8
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train = x_train[0:60,:]
#x_train = np.expand_dims(x_train, axis=3)

#model = tf.keras.Sequential([SE2Conv_Z2_H(filters=2, kernel_size=5, num_rotations=num_rotations, num_j=3, num_k=4, strides=1),
#                             SE2Conv_H_H(filters=2, kernel_size=5, num_rotations=num_rotations, num_j=3, num_k=4, strides=1),
#                             SE2Shift(num_rotations)])

#x_train = tf.convert_to_tensor(x_train/255., dtype=tf.float32)
#x_train_rot = tf.image.rot90(x_train)

#y = tf.reshape(model(x_train), [60, 28, 28, -1])
#y_rot = tf.reshape(model(x_train_rot), [60, 28, 28, -1])
#y_rot = tf.image.rot90(y_rot, 3)

#def show_images(y1, y2, idx):
#    for i in range(num_rotations):
#        plt.subplot(num_rotations, 2, 2*i+1)
#        plt.imshow(y1[idx, :, :, 2*i], cmap='jet')
#        plt.colorbar()        
#        plt.subplot(num_rotations, 2, 2*i+2)
#        plt.imshow(y2[idx, :, :, 2*i], cmap='jet')
#        plt.colorbar()
#    plt.savefig('haha.png')

#show_images(y, y_rot, 3)    





## debugging:
#num_rotations = 8
#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
#x_train = x_train[0:60,:]
#x_train = np.expand_dims(x_train, axis=3)

#model = tf.keras.Sequential([OE2Conv_Z2_H(filters=2, kernel_size=5, num_rotations=num_rotations, num_j=3, num_k=4, padding='same', strides=1),
#                             OE2Conv_H_H(filters=2, kernel_size=5, num_rotations=num_rotations, num_j=3, num_k=4, strides=1)])

#model = tf.keras.Sequential([OE2Conv_Z2_H(filters=2, kernel_size=5, num_rotations=num_rotations, num_j=3, num_k=4, padding='same', strides=1),
#                             OE2Conv_H_H_1x1(filters=2, num_rotations=num_rotations, strides=1)])

#model = tf.keras.Sequential([OE2Conv_Z2_H_1x1(filters=2, num_rotations=num_rotations, strides=1),
#                             OE2Conv_H_H_1x1(filters=2, num_rotations=num_rotations, strides=1)])


#x_train = tf.convert_to_tensor(x_train/255., dtype=tf.float32)
#x1 = x_train
#x2 = tf.image.flip_left_right(tf.image.rot90(x_train))

# for x1
#y1 = model(x1)
#z1 = OE2MaxPool(num_rotations)(y1)
#z1_transformed = tf.image.flip_left_right(tf.image.rot90(z1))

# for x2
#y2 = model(x2)
#z2 = OE2MaxPool(num_rotations)(y2)

# visualize

#def show_images(y1, y2):
#    for i in range(10):
#        plt.subplot(10, 2, 2*i+1)
#        plt.imshow(y1[i, :, :, 0], cmap='jet')
#        plt.colorbar()        
#        plt.subplot(10, 2, 2*i+2)
#        plt.imshow(y2[i, :, :, 0], cmap='jet')
#        plt.colorbar()
#    plt.savefig('haha.png')
    
#show_images(z1_transformed, z2)

