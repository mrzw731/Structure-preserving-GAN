"""
Contains architecture definitions for discriminator models.
"""

import numpy as np
import tensorflow.keras.layers as KL
from tensorflow_addons.layers import SpectralNormalization as SN
from tensorflow.keras import Model

from .blocks import discblock, ResBlockD
from .blocks_se2 import SE2_discblock, ResBlockD_OE2
from .layers import GroupMaxPool, GlobalSumPooling2D
from .layers_se2_more import SE2Shift, SE2MaxPool, OE2MaxPool
import pdb


def discriminator_model(img_shape, disc_arch='p4_food101', nclasses=6):
    """
    Return a tf image discriminator model.
    TODO: Redundant code, remember to clean up.

    Args:
        img_shape: tuple
            Input image shape.
        disc_arch: str
            Discriminator arch in format "x_y" where x in {"z2", "p4", "p4m"}
            and y in {"anhir128", "lysto128", "rotmnist", "cifar10", "food101"}
        nclasses: int
            Number of categories in the image set.
    """
    # Input images:
    input_img = KL.Input(shape=img_shape, name="main_input")

    if disc_arch == 'p4m_anhir128' or disc_arch == 'p4m_anhir64' or disc_arch == 'p4m_lysto64':
        sc = np.sqrt(8)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=int(16//sc), h_input='Z2',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(32//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(64//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(128//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(256//sc), h_input='D4',
                        h_output='D4', pad='same', stride=1)

        # Group pool:
        fea = KL.Activation('relu')(fea)
        fea = (GroupMaxPool('D4'))(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(256//sc)))(label)

    elif disc_arch == 'z2_anhir128' or disc_arch == 'z2_anhir64' or disc_arch == 'z2_lysto64':
        sc = np.sqrt(8)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=16, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=32, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=64, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=256, h_input='Z2',
                        h_output='Z2', pad='same', stride=1,
                        group_equiv=False)

        # Group pool:
        fea = KL.Activation('relu')(fea)
        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(256))(label)


    elif disc_arch == 'oe2_8_anhir64' or disc_arch == 'oe2_8_lysto64':
        sc = np.sqrt(8)

        # Define convolutional sequence:
        fea = ResBlockD_OE2(input_img, nfilters=int(16//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='Z2')
        fea = ResBlockD_OE2(fea, nfilters=int(32//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H')
        fea = ResBlockD_OE2(fea, nfilters=int(64//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H')
        fea = ResBlockD_OE2(fea, nfilters=int(128//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H')
        fea = ResBlockD_OE2(fea, nfilters=int(256//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H')
        
        # Group pool:
        fea = KL.Activation('relu')(fea)
        fea = OE2MaxPool(8)(fea)        

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(256//sc)))(label)


    elif disc_arch == 'se2_8_rotmnist':
        # mark to do
        # Define convolutional sequence:
        fea = SE2_discblock(input_img, nfilters=45, kernel_size=5, h_input='Z2',
                            num_rotations=8, num_j=3, num_k=3, pool='avg')
        fea = SE2_discblock(fea, nfilters=90, kernel_size=5, h_input='H',
                            num_rotations=8, num_j=3, num_k=3, pool='avg')
        fea = SE2_discblock(fea, nfilters=180, kernel_size=5, h_input='H',
                            num_rotations=8, num_j=3, num_k=3, pool='avg')        
        # Group pool D4 filters into Z2:
        fea = (SE2MaxPool(8))(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = KL.Dense(180)(label)


    elif disc_arch == 'se2_4_rotmnist':
        # mark to do
        # Define convolutional sequence:
        fea = SE2_discblock(input_img, nfilters=64, kernel_size=5, h_input='Z2',
                            num_rotations=4, num_j=3, num_k=3, pool='avg')
        fea = SE2_discblock(fea, nfilters=128, kernel_size=5, h_input='H',
                            num_rotations=4, num_j=3, num_k=3, pool='avg')
        fea = SE2_discblock(fea, nfilters=256, kernel_size=5, h_input='H',
                            num_rotations=4, num_j=3, num_k=3, pool='avg')        
        # Group pool D4 filters into Z2:
        fea = (SE2MaxPool(4))(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = KL.Dense(256)(label)

        
    elif disc_arch == 'p4_rotmnist':
        # Define convolutional sequence:
        fea = discblock(input_img, nfilters=64, h_input='Z2',
                        h_output='C4', pool='avg')
        fea = discblock(fea, nfilters=128, h_input='C4',
                        h_output='C4', pool='avg')
        fea = discblock(fea, nfilters=256, h_input='C4',
                        h_output='C4', pool='avg')
        # Group pool D4 filters into Z2:
        fea = (GroupMaxPool('C4'))(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = KL.Dense(256)(label)        

    elif disc_arch == 'z2_rotmnist':
        # Define convolutional sequence:
        fea = discblock(input_img, nfilters=128, h_input='Z2',
                        h_output='Z2', group_equiv=False, pool='avg')
        fea = discblock(fea, nfilters=256, h_input='Z2',
                        h_output='Z2', group_equiv=False, pool='avg')
        fea = discblock(fea, nfilters=512, h_input='Z2',
                        h_output='Z2', group_equiv=False, pool='avg')

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = KL.Dense(512)(label)

    elif disc_arch == 'z2_food101':
        sc = np.sqrt(4)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=128, h_input='Z2',
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=256, h_input='Z2',
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=512, h_input='Z2',
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=784, h_input='Z2',
                        h_output='Z2', group_equiv=False)

        fea = KL.Activation('relu')(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(784))(label)

    elif disc_arch == 'p4_food101':
        sc = np.sqrt(4)

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=int(128//sc), h_input='Z2',
                        h_output='C4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(256//sc), h_input='C4',
                        h_output='C4', pad='same', stride=1)
        fea = ResBlockD(fea, nfilters=int(512//sc), h_input='C4',
                        h_output='C4', pad='same', stride=1)
        fea = GroupMaxPool('C4')(fea)
        fea = ResBlockD(
            fea, nfilters=int(784), h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )

        fea = KL.Activation('relu')(fea)

        flat = KL.GlobalAveragePooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(784)))(label)

    elif disc_arch == 'p4_cifar10':
        sc = np.sqrt(4)

        fea = ResBlockD(input_img, nfilters=int(128//sc), h_input='Z2',
                        h_output='C4', poolshort='avg', poolskip='avg')
        fea = ResBlockD(fea, nfilters=int(128//sc), h_input='C4',
                        h_output='C4', poolshort='avg', poolskip='avg')
        fea = ResBlockD(fea, nfilters=int(128//sc), h_input='C4',
                        downsample=False, h_output='C4',
                        poolshort='avg', poolskip='avg')
        fea = GroupMaxPool('C4')(fea)
        fea = ResBlockD(fea, nfilters=int(128), h_input='Z2', downsample=False,
                        h_output='Z2', group_equiv=False,
                        poolshort='avg', poolskip='avg')

        fea = KL.Activation('relu')(fea)

        flat = GlobalSumPooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(int(128)))(label)

    elif disc_arch == 'z2_cifar10':

        # Define convolutional sequence:
        fea = ResBlockD(input_img, nfilters=128, h_input='Z2',
                        poolshort='avg', poolskip='avg', h_output='Z2',
                        group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2', poolshort='avg',
                        poolskip='avg', h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2', downsample=False,
                        h_output='Z2', group_equiv=False)
        fea = ResBlockD(fea, nfilters=128, h_input='Z2', downsample=False,
                        h_output='Z2', group_equiv=False)

        fea = KL.Activation('relu')(fea)

        flat = GlobalSumPooling2D()(fea)

        # Input class label:
        label = KL.Input(shape=(nclasses,))
        label_emb = SN(KL.Dense(128))(label)

    # Projection discriminator:
    projection = KL.dot([flat, label_emb], axes=1)
    op = SN(KL.Dense(1))(flat)

    prediction = KL.Add()([projection, op])
    model = Model(inputs=[input_img, label], outputs=prediction)

    return model
