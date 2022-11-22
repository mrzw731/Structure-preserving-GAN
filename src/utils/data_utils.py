import numpy as np
from tensorflow.keras.utils import to_categorical
import pdb
import matplotlib.pyplot as plt

def npy_loader(dataset, num_classes, num_samples=None):
    """
    Utility function to load npy files corresponding to training images and
    labels.
    Args:
        dataset: str
            Name of dataset. One of {'anhir128', 'lysto128', cifar10', 'food101',
            'rotmnist'}.
        num_classes: int
            Number of categories in the image set.
    """

    data = np.load('./data/{}/train_images.npy'.format(dataset))
    labels = np.load('./data/{}/train_labels.npy'.format(dataset))
    labels = to_categorical(labels, num_classes=num_classes)
    if num_samples is not None:
        rng = np.random.default_rng()
        indx=rng.choice(labels.shape[0],num_samples)
        data=data[indx,:,:,:]
        labels=labels[indx]
    
    print(data.shape)
    print(labels.shape)
    return data, labels


def dataset_lookup(dataset):
    """
    Utility function to return the number of classes for a chosen dataset.
    Args:
        dataset: str
            Name of dataset. One of {'anhir128', 'lysto128', cifar10', 'food101',
            'rotmnist'}
    """
    if dataset == 'anhir128':
        num_classes = 5
        print_multiplier = 8
    elif dataset == 'anhir64':
        num_classes = 5
        print_multiplier = 8        
    elif dataset == 'lysto128' or dataset == 'lysto64':
        num_classes = 3
        print_multiplier = 8
    elif dataset == 'cifar10':
        num_classes = 10
        print_multiplier = 1
    elif dataset == 'rotmnist':
        num_classes = 10
        print_multiplier = 1
    elif dataset == 'food101':
        num_classes = 101
        print_multiplier = 1
    else:
        raise ValueError
    return num_classes, print_multiplier


def data_normalizer(tensor, dataset):
    """
    Rescale intensities of the input dataset.
    Args:
        tensor: np array
            Batch to rescale.
        dataset: str
            Name of dataset. One of {'anhir128', 'lysto128', cifar10', 'food101',
            'rotmnist'}
    """
    if dataset == 'rotmnist':
        return (tensor * 2.0) - 1.0
    else:
        return (tensor/127.5) - 1.0


def data_generator(data, labels, batch_size, noise_dim, dataset, aug):
    """
    Numpy data generator for GAN training.
    Args:
        data: np array
            Overall image data array to sample from.
        labels: np array
            Overall label array to sample from.
        batch_size: int
            Batch size to return.
        noise_dim:
            Latent dimensionality.
        dataset: str
            Name of dataset. One of {'anhir128', 'lysto128', cifar10', 'food101',
            'rotmnist'}
        aug: 0 or 1
            Without (0) or with (1) p4m data augmentation
    """
    datasize = data.shape[0]
    while True:
        image_idx = np.random.randint(0, datasize, batch_size)

        noise = np.random.randn(batch_size, noise_dim).astype(np.float32)
        
        # Add two more dimension for noise corresponding to rotation and reflection
        # This is used to implement the correct generator
        
        more_noise = np.random.uniform(size=[batch_size, 2]).astype(np.float32)
        noise = np.hstack([noise, more_noise])
        
        real_images_batch = data_normalizer(data[image_idx], dataset)
        if aug == 1:
            real_images_batch_flip = np.flip(real_images_batch, axis=1)
            
            aug_noise = np.random.uniform(size=[batch_size, 2]).astype(np.float32)
            rot_angles = aug_noise[:, 0]/.25
            rot_angles = rot_angles.astype(int)
            
            is_flip = aug_noise[:, 1]/.5
            is_flip = is_flip.astype(int)

            out_images_batch = np.zeros_like(real_images_batch)

            for k in np.arange(4):
                #idx_0 = (rot_angles ==k) & (is_flip == 0)
                #idx_0 = idx_0.astype(int)
                #idx_0 = idx_0.reshape([-1, 1, 1, 1])
                #out_images_batch = out_images_batch + np.rot90(real_images_batch*idx_0, k=k, axes=(1, 2))

                #idx_1 = (rot_angles ==k) & (is_flip == 1)
                #idx_1 = idx_1.astype(int)
                #idx_1 = idx_1.reshape([-1, 1, 1, 1])
                #out_images_batch = out_images_batch + np.rot90(real_images_batch_flip*idx_1, k=k, axes=(1, 2))

                idx_0 = (rot_angles ==k) & (is_flip == 0)
                out_images_batch[idx_0, ...] = np.rot90(real_images_batch[idx_0, ...], k=k, axes=(1, 2))
                idx_1 = (rot_angles ==k) & (is_flip == 1)
                out_images_batch[idx_1, ...] = np.rot90(real_images_batch_flip[idx_1, ...], k=k, axes=(1, 2))

            real_images_batch = out_images_batch

        real_labels_batch = labels[image_idx]

        yield noise, real_images_batch, real_labels_batch


def generator_dimensionality(gen_arch):
    """
    Return dimensionalities used in paper corresponding to the datasets chosen.

    Args:
        gen_arch: str
            Generator arch fmt "x_y" where x in {"z2", "p4", "p4m"}
            and y in {"anhir128", "lysto128", "rotmnist", "cifar10", "food101"}
    """
    # To keep the number of parameters across settings roughly consistent:
    if 'p4_' in gen_arch:
        channel_scale_factor = np.sqrt(4)
    if 'p4m_' in gen_arch:
        channel_scale_factor = np.sqrt(8)
    if 'z2_' in gen_arch:
        channel_scale_factor = 1
    if 'se2_4_' in gen_arch:
        channel_scale_factor = np.sqrt(4)
    if 'se2_8_' in gen_arch:
        channel_scale_factor = np.sqrt(8)
    if 'oe2_8_' in gen_arch:
        channel_scale_factor = np.sqrt(8)                

    if '_anhir128' in gen_arch:
        projection_dimensionality = 128 * 4 * 4  # latent linear projection
        projection_reshape = (4, 4, 128)
        label_emb_dim = 128  # Embedding dimension for one-hot label vector.
    elif '_anhir64' in gen_arch:
        projection_dimensionality = 128 * 4 * 4  # latent linear projection
        projection_reshape = (4, 4, 128)
        label_emb_dim = 128  # Embedding dimension for one-hot label vector.        
    elif '_lysto128' in gen_arch:
        projection_dimensionality = 128 * 4 * 4
        projection_reshape = (4, 4, 128)
        label_emb_dim = 128
    elif '_lysto64' in gen_arch:
        projection_dimensionality = 128 * 4 * 4
        projection_reshape = (4, 4, 128)
        label_emb_dim = 128        
    elif '_cifar10' in gen_arch:
        projection_dimensionality = 256 * 4 * 4
        projection_reshape = (4, 4, 256)
        label_emb_dim = 128
    elif '_rotmnist' in gen_arch:
        projection_dimensionality = 128 * 7 * 7
        projection_reshape = (7, 7, 128)
        label_emb_dim = 64
    elif '_food101' in gen_arch:
        projection_dimensionality = 1024 * 4 * 4
        projection_reshape = (4, 4, 1024)
        label_emb_dim = 64
    else:
        raise ValueError

    return (
        channel_scale_factor,
        projection_dimensionality,
        projection_reshape,
        label_emb_dim
    )
