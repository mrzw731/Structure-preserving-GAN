"""
Contains architecture definitions for generator models.
"""

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.layers as KL
from tensorflow.keras.models import Model
from tensorflow_addons.layers import SpectralNormalization as SN

from keras_gcnn.layers import GConv2D, GBatchNorm
from .layers_se2_conv import SE2Conv_Z2_H, SE2Conv_H_H, OE2Conv_H_H

from .utils.data_utils import generator_dimensionality
from .layers import GShift, GroupMaxPool
from .blocks import CCBN, ResBlockG_film
from .layers_se2_more import SE2Shift, SE2MaxPool, OE2MaxPool
from .blocks_se2 import SE2_CCBN, SE2_symmetrize, OE2_symmetrize, ResBlock_OE2_film, P4_symmetrize, P4m_symmetrize
import pdb


def generator_model(nclasses=101, gen_arch='p4_food101', latent_dim=64):
    """
    Return a tf image generator model.

    Args:
        nclasses: int
            Number of categories in the image set.
        gen_arch: str
            Generator arch in format "x_y" where x in {"z2", "p4", "p4m", "correct_p4", "correct_p4m"}
            and y in {"anhir128", "lysto128", "rotmnist", "cifar10", "food101"}
        latent_dim: int
            Dimensionality of Gaussian latents.
    """

    # Get latent vector:
    # Add two more dimension for noise corresponding to rotation and reflection
    # This is used to implement the correct generator    
    latent_vec_all = KL.Input(shape=(latent_dim+2,))
    latent_vec = latent_vec_all[:, :-2]
    latent_vec_rot_ref = latent_vec_all[:, -2:]
    label_vec = KL.Input(shape=(nclasses,))

    sc, proj_dim, proj_shape, labelemb_dim = generator_dimensionality(gen_arch)
    label_proj = KL.Dense(  # Using SN here seems to lead to collapse
        labelemb_dim,
        use_bias=False,
    )(label_vec)

    # Concatenate noise and condition feature maps to modulate the generator:
    cla = KL.concatenate([latent_vec, label_proj])

    # Project and reshape to spatial feature maps:
    gen = SN(KL.Dense(proj_dim))(cla)
    gen = KL.Reshape(proj_shape)(gen)



    if gen_arch == 'cutcorner_lvl2_oe2_8_anhir64':
        # rotating and reflecting indices
        rot_angles = latent_vec_rot_ref[:, -2]/.125
        rot_angles = tf.cast(rot_angles, tf.int32)

        is_flip = latent_vec_rot_ref[:, -1]/.5
        is_flip = tf.cast(is_flip, tf.int32)

        # One level of Z2 residual convolutions + Upsampling
        fea = ResBlockG_film(
            gen, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )

        # Cutcorner OE2 symmetrizing with 8 rotations

        fea = OE2_symmetrize(fea, rot_angles, is_flip, 8, True)

        # Another level of H residual convolutions + Upsampling
        
        fea = ResBlock_OE2_film(
            fea, cla, int(128//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='Z2',
        )
        for ch in [64, 32]:
            fea = ResBlock_OE2_film(
                fea, cla, int(ch//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H',
            )

        # last one not upsampling
        fea = ResBlock_OE2_film(
            fea, cla, int(16//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H', upsample=False,
        )

        
        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(OE2Conv_H_H(3, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        op = OE2MaxPool(8)(fea)        


    elif gen_arch == 'not_cutcorner_lvl1_oe2_8_anhir64':
        # rotating and reflecting indices
        rot_angles = latent_vec_rot_ref[:, -2]/.25
        rot_angles = tf.cast(rot_angles, tf.int32)

        is_flip = latent_vec_rot_ref[:, -1]/.5
        is_flip = tf.cast(is_flip, tf.int32)
        # OE2 symmetrizing without cutting corners
        fea = OE2_symmetrize(gen, rot_angles, is_flip, 4, False)
        
        # Another level of H residual convolutions + Upsampling


        fea = ResBlock_OE2_film(
            fea, cla, int(256//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='Z2',
        )
        
        for ch in [128, 64, 32]:
            fea = ResBlock_OE2_film(
                fea, cla, int(ch//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H',
            )

        # last one not upsampling
        fea = ResBlock_OE2_film(
            fea, cla, int(16//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H', upsample=False,
        )

        
        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(OE2Conv_H_H(3, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        op = OE2MaxPool(8)(fea)        
        

        
    elif gen_arch == 'correct_p4m_anhir64':
        # rotating and reflecting first
        rot_angles = latent_vec_rot_ref[:, -2]/.25
        rot_angles = tf.cast(rot_angles, tf.int32)

        is_flip = latent_vec_rot_ref[:, -1]/.5
        is_flip = tf.cast(is_flip, tf.int32)

        fea = P4m_symmetrize(gen, rot_angles, is_flip)
        #fea = OE2_symmetrize(gen, rot_angles, is_flip, 4, False)        

        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            fea, cla, int(256//sc), h_input='Z2', h_output='D4', pad='same',
        )
        for ch in [128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='D4', h_output='D4', pad='same',
            )

        # last one not upsampling
        fea = ResBlockG_film(
            fea, cla, int(16//sc), h_input='D4', h_output='D4', pad='same', upsample=False,
        )
        

        fea = GBatchNorm(h='D4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='D4', h_output='D4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('D4')(fea)        
        

    elif gen_arch == 'p4m_anhir64':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(256//sc), h_input='Z2', h_output='D4', pad='same',
        )
        for ch in [128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='D4', h_output='D4', pad='same',
            )

        # last one not upsampling            
        fea = ResBlockG_film(
            fea, cla, int(16//sc), h_input='D4', h_output='D4', pad='same', upsample=False,
            )

        
        fea = GBatchNorm(h='D4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='D4', h_output='D4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('D4')(fea)

    elif gen_arch == 'z2_anhir64':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )
        for ch in [128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, ch, h_input='Z2', h_output='Z2',
                pad='same', group_equiv=False,
            )

        # last one not upsampling
        fea = ResBlockG_film(
            fea, cla, 16, h_input='Z2', h_output='Z2',
            pad='same', group_equiv=False, upsample=False,
        )
        

        fea = KL.BatchNormalization(
            momentum=0.1, center=False, scale=False,
        )(fea)
        fea = CCBN(fea, cla, 'Z2')
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(
            3, kernel_size=3, padding='same', use_bias=False,
        ))(fea)



    elif gen_arch == 'cutcorner_lvl2_oe2_8_lysto64':
        # rotating and reflecting indices
        rot_angles = latent_vec_rot_ref[:, -2]/.125
        rot_angles = tf.cast(rot_angles, tf.int32)

        is_flip = latent_vec_rot_ref[:, -1]/.5
        is_flip = tf.cast(is_flip, tf.int32)

        # One level of Z2 residual convolutions + Upsampling
        fea = ResBlockG_film(
            gen, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )

        # Cutcorner OE2 symmetrizing with 8 rotations

        fea = OE2_symmetrize(fea, rot_angles, is_flip, 8, True)

        # Another level of H residual convolutions + Upsampling
        
        fea = ResBlock_OE2_film(
            fea, cla, int(128//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='Z2',
        )
        for ch in [64, 32]:
            fea = ResBlock_OE2_film(
                fea, cla, int(ch//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H',
            )

        # last one not upsampling
        fea = ResBlock_OE2_film(
            fea, cla, int(16//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H', upsample=False,
        )

        
        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(OE2Conv_H_H(3, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        op = OE2MaxPool(8)(fea)        


    elif gen_arch == 'not_cutcorner_lvl1_oe2_8_lysto64':
        # rotating and reflecting indices
        rot_angles = latent_vec_rot_ref[:, -2]/.25
        rot_angles = tf.cast(rot_angles, tf.int32)

        is_flip = latent_vec_rot_ref[:, -1]/.5
        is_flip = tf.cast(is_flip, tf.int32)
        # OE2 symmetrizing without cutting corners
        fea = OE2_symmetrize(gen, rot_angles, is_flip, 4, False)
        
        # Another level of H residual convolutions + Upsampling


        fea = ResBlock_OE2_film(
            fea, cla, int(256//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='Z2',
        )
        
        for ch in [128, 64, 32]:
            fea = ResBlock_OE2_film(
                fea, cla, int(ch//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H',
            )

        # last one not upsampling
        fea = ResBlock_OE2_film(
            fea, cla, int(16//sc), kernel_size=5, num_rotations=8, num_j=3, num_k=3, h_input='H', upsample=False,
        )

        
        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(OE2Conv_H_H(3, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        op = OE2MaxPool(8)(fea)        
        

        
    elif gen_arch == 'correct_p4m_lysto64':
        # rotating and reflecting first
        rot_angles = latent_vec_rot_ref[:, -2]/.25
        rot_angles = tf.cast(rot_angles, tf.int32)

        is_flip = latent_vec_rot_ref[:, -1]/.5
        is_flip = tf.cast(is_flip, tf.int32)

        fea = P4m_symmetrize(gen, rot_angles, is_flip)
        #fea = OE2_symmetrize(gen, rot_angles, is_flip, 4, False)        

        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            fea, cla, int(256//sc), h_input='Z2', h_output='D4', pad='same',
        )
        for ch in [128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='D4', h_output='D4', pad='same',
            )

        # last one not upsampling
        fea = ResBlockG_film(
            fea, cla, int(16//sc), h_input='D4', h_output='D4', pad='same', upsample=False,
        )
        

        fea = GBatchNorm(h='D4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='D4', h_output='D4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('D4')(fea)        
        

    elif gen_arch == 'p4m_lysto64':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(256//sc), h_input='Z2', h_output='D4', pad='same',
        )
        for ch in [128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='D4', h_output='D4', pad='same',
            )

        # last one not upsampling            
        fea = ResBlockG_film(
            fea, cla, int(16//sc), h_input='D4', h_output='D4', pad='same', upsample=False,
            )

        
        fea = GBatchNorm(h='D4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='D4', h_output='D4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('D4')(fea)

    elif gen_arch == 'z2_lysto64':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )
        for ch in [128, 64, 32]:
            fea = ResBlockG_film(
                fea, cla, ch, h_input='Z2', h_output='Z2',
                pad='same', group_equiv=False,
            )

        # last one not upsampling
        fea = ResBlockG_film(
            fea, cla, 16, h_input='Z2', h_output='Z2',
            pad='same', group_equiv=False, upsample=False,
        )
        

        fea = KL.BatchNormalization(
            momentum=0.1, center=False, scale=False,
        )(fea)
        fea = CCBN(fea, cla, 'Z2')
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(
            3, kernel_size=3, padding='same', use_bias=False,
        ))(fea)


    elif gen_arch == 'cutcorner_lvl1_se2_8_rotmnist':
        # rotating first
        rot_angles = latent_vec_rot_ref[:, -2]/.125
        rot_angles = tf.cast(rot_angles, tf.int32)                
        
        fea = SE2_symmetrize(gen, rot_angles, 8, True)
                
        # Convolutions + Upsampling:
        fea = SN(SE2Conv_Z2_H(180, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        fea = SE2Shift(8)(fea)
        fea = KL.Activation('relu')(fea)
        B, H, W, _, num_channels = fea.shape
        fea = KL.Reshape((H, W, 8*num_channels))(fea)
        fea = KL.UpSampling2D()(fea)
        fea = KL.Reshape((2*H, 2*W, 8, num_channels))(fea)        
        fea = SN(SE2Conv_H_H(90, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = SE2_CCBN(fea, cla, 8)
        fea = KL.Activation('relu')(fea)

        fea = KL.Reshape((14, 14, 8*90))(fea)        
        fea = KL.UpSampling2D()(fea)
        fea = KL.Reshape((28, 28, 8, 90))(fea)                
        fea = SN(SE2Conv_H_H(45, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = SE2_CCBN(fea, cla, 8)
        fea = KL.Activation('relu')(fea)

        fea = SN(SE2Conv_H_H(1, kernel_size=5, num_rotations=8, num_j=3, num_k=3))(fea)
        op = SE2MaxPool(8)(fea)

        
    ## new architecture: correct p4_rotmnist generator.
    elif gen_arch == 'correct_p4_rotmnist':
        # rotating first
        rot_angles = latent_vec_rot_ref[:, -2]/.25
        rot_angles = tf.cast(rot_angles, tf.int32)                

        fea = P4_symmetrize(gen, rot_angles)
        #fea = SE2_symmetrize(gen, rot_angles, 4, False)
            
        # Convolutions + Upsampling:
        fea = SN(GConv2D(256, kernel_size=3, h_input='Z2',
                         h_output='C4', padding='same'))(fea)
        fea = GShift(h='C4')(fea)
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(GConv2D(128, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        fea = GBatchNorm(h='C4', momentum=0.1, center=False, scale=False)(fea)
        fea = CCBN(fea, cla, 'C4')
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(GConv2D(64, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        fea = GBatchNorm(h='C4', momentum=0.1, center=False, scale=False)(fea)
        fea = CCBN(fea, cla, 'C4')
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(1, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        op = GroupMaxPool('C4')(fea)

        
    elif gen_arch == 'p4_rotmnist':
        # Convolutions + Upsampling:
        fea = SN(GConv2D(256, kernel_size=3, h_input='Z2',
                         h_output='C4', padding='same'))(gen)
        fea = GShift(h='C4')(fea)
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(GConv2D(128, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        fea = GBatchNorm(h='C4', momentum=0.1, center=False, scale=False)(fea)
        fea = CCBN(fea, cla, 'C4')
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(GConv2D(64, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        fea = GBatchNorm(h='C4', momentum=0.1, center=False, scale=False)(fea)
        fea = CCBN(fea, cla, 'C4')
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(1, kernel_size=3, h_input='C4',
                         h_output='C4', padding='same'))(fea)
        op = GroupMaxPool('C4')(fea)

    elif gen_arch == 'z2_rotmnist':
        # Convolutions + Upsampling:
        fea = SN(KL.Conv2D(512, kernel_size=3, padding='same',
                           use_bias=True,
                           kernel_initializer='orthogonal'))(gen)
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(KL.Conv2D(256, kernel_size=3, padding='same',
                           use_bias=False,
                           kernel_initializer='orthogonal'))(fea)
        fea = KL.BatchNormalization(
            momentum=0.1, center=False, scale=False,
        )(fea)
        fea = CCBN(fea, cla, 'Z2')
        fea = KL.Activation('relu')(fea)

        fea = KL.UpSampling2D()(fea)
        fea = SN(KL.Conv2D(128, kernel_size=3, padding='same',
                           use_bias=False,
                           kernel_initializer='orthogonal'))(fea)
        fea = KL.BatchNormalization(
            momentum=0.1, center=False, scale=False,
        )(fea)
        fea = CCBN(fea, cla, 'Z2')
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(1, kernel_size=3, padding='same', use_bias=False,
                          kernel_initializer='orthogonal'))(fea)

    elif gen_arch == 'p4_food101':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(512//sc), h_input='Z2', h_output='C4', pad='same',
        )
        for ch in [384, 256, 192]:
            fea = ResBlockG_film(
                fea, cla, int(ch//sc), h_input='C4', h_output='C4', pad='same',
            )

        fea = GBatchNorm(h='C4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='C4', h_output='C4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('C4')(fea)

    elif gen_arch == 'z2_food101':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(gen, cla, 512, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)
        fea = ResBlockG_film(fea, cla, 384, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)
        fea = ResBlockG_film(fea, cla, 256, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)
        fea = ResBlockG_film(fea, cla, 192, h_input='Z2',
                             h_output='Z2', pad='same', group_equiv=False)

        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(3, kernel_size=3, padding='same',
                          kernel_initializer='orthogonal'))(fea)

    elif gen_arch == 'p4_cifar10':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, int(256//sc), h_input='Z2', h_output='C4', pad='same',
        )
        fea = ResBlockG_film(
            fea, cla, int(256//sc), h_input='C4', h_output='C4', pad='same',
        )
        fea = ResBlockG_film(
            fea, cla, int(256//sc), h_input='C4', h_output='C4', pad='same',
        )

        fea = GBatchNorm(h='C4', momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        fea = SN(GConv2D(3, kernel_size=3, h_input='C4', h_output='C4',
                         padding='same', kernel_initializer='orthogonal'))(fea)
        op = GroupMaxPool('C4')(fea)

    elif gen_arch == 'z2_cifar10':
        # (Residual) Convolutions + Upsampling:
        fea = ResBlockG_film(
            gen, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', group_equiv=False,
        )
        fea = ResBlockG_film(
            fea, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )
        fea = ResBlockG_film(
            fea, cla, 256, h_input='Z2', h_output='Z2',
            pad='same', stride=1, group_equiv=False,
        )

        fea = KL.BatchNormalization(momentum=0.1)(fea)
        fea = KL.Activation('relu')(fea)

        op = SN(KL.Conv2D(
            3, kernel_size=3, padding='same', use_bias=False,
        ))(fea)

    else:
        raise ValueError('Generator Architecture Unrecognized')

    gen_img = KL.Activation('tanh')(op)  # Get final synthesized image batch

    # Generator model:
    generator = Model([latent_vec_all, label_vec], gen_img)

    return generator
