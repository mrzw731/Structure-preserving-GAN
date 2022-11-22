"""
Main training script for Structure-preserving GAN using Lipschitz-alpha loss

Author: Jeremiah Birrell, Markos A. Katsoulakis, Luc Rey-Bellet, Wei Zhu
"""

import datetime
import os
import random
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm

from time import time
from tensorflow.compat.v1 import set_random_seed
from tensorflow.keras.utils import Progbar

from src.discriminators import discriminator_model
from src.generators import generator_model
from src.optimizers import get_optimizers
from src.utils.data_utils import dataset_lookup, npy_loader, data_generator
from src.utils.training_args import training_args
from src.utils.fid import calculate_activation_statistics, calculate_frechet_distance

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pdb
#tf.config.run_functions_eagerly(True)


# Parse CLI:
args = training_args()


num_classes, print_multiplier = dataset_lookup(args.dataset)

if args.dataset == 'rotmnist':
    fid_cal_multiplier = (10000 // (num_classes * num_classes)) + 1
elif args.dataset == 'anhir128':
    fid_cal_multiplier = (75752 // (num_classes * num_classes)) + 1
elif args.dataset == 'anhir64':
    fid_cal_multiplier = (75752 // (num_classes * num_classes)) + 1    
elif args.dataset == 'lysto128' or args.dataset == 'lysto64':
    fid_cal_multiplier = (54000 // (num_classes * num_classes)) + 1

    
# Load FID model and statistics for dataset

if args.dataset == 'rotmnist':
    inception_model = tf.keras.models.load_model('fid/fid_models/rotmnist/autoencoder')
    inception_model = inception_model.encoder
    activation_dim = 64    
    inception_model.summary()
    with np.load('fid/fid_statistics/'+args.dataset+'.npz') as fid_stats_real:
        mu_real = fid_stats_real['mu_real']
        sigma_real = fid_stats_real['sigma_real']
        
elif args.dataset == 'anhir128':
    inception_model = tf.keras.models.load_model('fid/fid_models/anhir128/inceptionv3_feature')
    activation_dim = 2048
    inception_model.summary()
    with np.load('fid/fid_statistics/anhir128.npz') as fid_stats_real:
        mu_real = fid_stats_real['mu_real']
        sigma_real = fid_stats_real['sigma_real']

elif args.dataset == 'anhir64':
    inception_model = tf.keras.models.load_model('fid/fid_models/anhir64/inceptionv3_feature')
    activation_dim = 2048
    inception_model.summary()
    with np.load('fid/fid_statistics/anhir64.npz') as fid_stats_real:
        mu_real = fid_stats_real['mu_real']
        sigma_real = fid_stats_real['sigma_real']

        
elif args.dataset == 'lysto128':
    inception_model = tf.keras.models.load_model('fid/fid_models/lysto128/inceptionv3_feature')
    activation_dim = 2048
    inception_model.summary()
    with np.load('fid/fid_statistics/lysto128.npz') as fid_stats_real:
        mu_real = fid_stats_real['mu_real']
        sigma_real = fid_stats_real['sigma_real']

elif args.dataset == 'lysto64':
    inception_model = tf.keras.models.load_model('fid/fid_models/lysto64/inceptionv3_feature')
    activation_dim = 2048
    inception_model.summary()
    with np.load('fid/fid_statistics/lysto64.npz') as fid_stats_real:
        mu_real = fid_stats_real['mu_real']
        sigma_real = fid_stats_real['sigma_real']


# Set RNG for numpy and tensorflow
np.random.seed(args.rng)
set_random_seed(args.rng)
random.seed(args.rng)

# Set format for directory names to save models in:
save_folder = (('{}_{}_{}epochs_Garch_{}_Darch_{}'
                '_dupdates{}_lrg{}_lrd{}_gp{}_batchsize{}_alpha{}_rev{}_aug{}_run{}_num_sample{}')
               .format(
                   args.name, args.dataset, args.epochs,
                   args.g_arch, args.d_arch, args.d_updates, args.lr_g,
                   args.lr_d, args.gp_wt, args.batchsize, args.alpha, args.reverse, args.aug, args.run, args.num_samples
))


# ---------------------------------------------------------------------------
# Data loading

# Load dataset:
data, labels = npy_loader(args.dataset, num_classes, args.num_samples)

# Set up data generator:
datagen = data_generator(
    data, labels, args.batchsize, args.latent_dim, args.dataset, args.aug
)

# ---------------------------------------------------------------------------
# Intialize networks

# Define generator and discriminator networks:
generator = generator_model(
    nclasses=num_classes, gen_arch=args.g_arch, latent_dim=args.latent_dim,
)

discriminator = discriminator_model(
    img_shape=data.shape[1:], nclasses=num_classes, disc_arch=args.d_arch,
)

# Create optimizers:
goptim, doptim = get_optimizers(
    args.lr_g, args.beta1_g, args.beta2_g,  # generator adam params
    args.lr_d, args.beta1_d, args.beta2_d,  # discriminator adam params
)


# ---------------------------------------------------------------------------
# Plotting and checkpointing setup:

# Setup training checkpoints:
checkpoint_dir = './training_checkpoints/{}'.format(save_folder)
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=goptim,
    discriminator_optimizer=doptim,
    generator=generator,
    discriminator=discriminator,
)

# Location to save training samples:
os.makedirs('training_pngs/' + save_folder, exist_ok=True)

# Parameters for samples visualized and saved as PNGs:
test_labels = np.transpose(
    np.tile(np.eye(num_classes), num_classes*print_multiplier),
)
test_labels_fid = np.transpose(
    np.tile(np.eye(num_classes), num_classes*fid_cal_multiplier),
)

# If resuming training
if args.resume_ckpt > 0:
    checkpoint.restore(
        './training_checkpoints/{}/ckpt-{}'.format(
            save_folder, args.resume_ckpt,
        )
    ).assert_consumed()

summary_writer = tf.summary.create_file_writer(
    'training_logs/{}'.format(save_folder) +
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
)


# ---------------------------------------------------------------------------
# Training loops

EPS = tf.convert_to_tensor(1e-6, dtype=tf.float32)


#for alpha>1
alpha=args.alpha
def f_alpha_star(y):
    return tf.math.pow(tf.nn.relu(y),alpha/(alpha-1.0))*tf.math.pow((alpha-1.0),alpha/(alpha-1.0))/alpha+1/(alpha*(alpha-1.0))



@tf.function
def gen_train_step(
    real_batch, label, noise_batch, step, eps=EPS,
):
    """
    Generator training step. Only supports the relativistic average loss for
    now.
    # TODO: abstract out loss and support more types of losses.
    Args:
        real_batch: np.array (batch_size, x, y, ch)
            Batch of randomly sampled real images.
        label: (batch_size, n_classes)
            Batch of labels corresponding to the randomly sampled reals above.
        noise_batch: np.array (batch_size, latent_dim + 2)
            Batch of random latents. The last two dims correspond to rotation + reflection
        eps: tf float
            Constant to keep the log function happy.
    """

    with tf.GradientTape() as gen_tape:
        # Generate fake images to feed discriminator:
        fake_batch = generator([noise_batch, label], training=True)

        # Get discriminator logits on real images and fake images:
        # Could use different labels for fakes too. Doesn't make a
        # noticeable difference.
        disc_opinion_real = discriminator([real_batch, label], training=True)
        disc_opinion_fake = discriminator([fake_batch, label], training=True)

        reverse=args.reverse
        if reverse == 1:
            gen_cost_fake = tf.reduce_mean(disc_opinion_fake)
            gen_cost_real = tf.reduce_mean(f_alpha_star(disc_opinion_real))
            gen_loss=gen_cost_fake-gen_cost_real
        elif reverse == 0:
            gen_cost_real = tf.reduce_mean(disc_opinion_real)
            gen_cost_fake = tf.reduce_mean(f_alpha_star(disc_opinion_fake))
            gen_loss=gen_cost_real-gen_cost_fake

     
    # Get gradients and update generator:
    generator_gradients = gen_tape.gradient(
        gen_loss, generator.trainable_variables,
    )

    goptim.apply_gradients(
        zip(generator_gradients, generator.trainable_variables),
    )

    with summary_writer.as_default():
        tf.summary.scalar('losses/g_loss', gen_loss, step=step)
        tf.summary.image('samples', 0.5*(fake_batch + 1), step=step)


@tf.function
def disc_train_step(
    real_batch, label, noise_batch, step, eps=EPS,
):
    """
    Discriminator training step. So far only supports the relavg_gp loss.
    # TODO: abstract out loss and support more types of losses.
    Args:
        real_batch: np.array (batch_size, x, y, ch)
            Batch of randomly sampled real images.
        label: (batch_size, n_classes)
            Batch of labels corresponding to the randomly sampled reals above.
        noise_batch: np.array (batch_size, latent_dim + 2)
            Batch of random latents. The last two dims correspond to rotation + reflection
        eps: tf float
            Constant to keep the log function happy.
    """
    gp_strength = tf.constant(args.gp_wt, dtype=tf.float32)

    with tf.GradientTape() as disc_tape:
        # Generate fake images to feed discriminator:
        fake_batch = generator([noise_batch, label], training=True)

        # Get discriminator logits on real images and fake images:
        # Could use different labels for fakes too. Doesn't make a
        # noticeable difference.
        disc_opinion_real = discriminator([real_batch, label], training=True)
        disc_opinion_fake = discriminator([fake_batch, label], training=True)

        reverse=args.reverse
        if reverse == 1:
            disc_cost_fake = tf.reduce_mean(disc_opinion_fake)
            disc_cost_real = tf.reduce_mean(f_alpha_star(disc_opinion_real))
            disc_loss=disc_cost_fake-disc_cost_real
        elif reverse == 0:
            disc_cost_real = tf.reduce_mean(disc_opinion_real)
            disc_cost_fake = tf.reduce_mean(f_alpha_star(disc_opinion_fake))
            disc_loss=disc_cost_real-disc_cost_fake

        

      
        # Get gradient penalty:
        new_real_batch = 1.0 * real_batch
        new_fake_batch = 1.0 * fake_batch
        new_label = 1.0 * label
        eta = tf.random.uniform(shape=[args.batchsize,1,1,1], minval=0., maxval=1.)
        differences = new_fake_batch - new_real_batch
        interpolates = new_real_batch + (eta*differences)
        

        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolates)
            disc_opinion_interp = discriminator(
                [interpolates, new_label], training=True,
            )

        grad = gp_tape.gradient(disc_opinion_interp, interpolates)
        grad_sqr = tfm.square(grad)
        grad_sqr_sum = tf.reduce_sum(
                grad_sqr,
                axis=np.arange(1, len(grad_sqr.shape)),
        )
        gradient_penalty = gp_strength * tf.reduce_mean(tf.math.maximum(0.,grad_sqr_sum-1.0))
        total_disc_loss = -disc_loss + gradient_penalty

    # Get gradients and update discriminator:
    discriminator_gradients = disc_tape.gradient(
        total_disc_loss,
        discriminator.trainable_variables,
    )

    doptim.apply_gradients(
        zip(discriminator_gradients, discriminator.trainable_variables),
    )

    with summary_writer.as_default():
        tf.summary.scalar('losses/d_loss', disc_loss, step=step)
        tf.summary.scalar('regularizers/GP', gradient_penalty, step=step)


# ---------------------------------------------------------------------------
# Train loop:

gen_update_count = 0
fids = []

for epoch in range(args.epochs):
    print("epoch {} of {}".format(epoch + 1, args.epochs))
    nbatches = args.batchsize * (args.d_updates + 1)

    # Print progress bar:
    progress_bar = Progbar(target=int(data.shape[0] // nbatches))

    # Loop through each batch:
    start_time = time()
    steps = int(data.shape[0] // nbatches)
    for index in range(steps):  # Loop through steps
        progress_bar.update(index)

        # Update discriminator:
        for j in range(args.d_updates):
            noise, image_batch, labs_batch = next(iter(datagen))

            disc_train_step(
                tf.convert_to_tensor(image_batch, dtype=tf.float32),
                tf.convert_to_tensor(labs_batch, dtype=tf.float32),
                tf.convert_to_tensor(noise, dtype=tf.float32),
                tf.convert_to_tensor((index + epoch*steps), dtype=tf.int64),
            )

        # Update Generator:
        noise, image_batch, labs_batch = next(iter(datagen))

        gen_train_step(
            tf.convert_to_tensor(image_batch, dtype=tf.float32),
            tf.convert_to_tensor(labs_batch, dtype=tf.float32),
            tf.convert_to_tensor(noise, dtype=tf.float32),
            tf.convert_to_tensor((index + epoch*steps), dtype=tf.int64),
        )

        # Keep track of generator update count and calculate FID if necessary
        
        gen_update_count = gen_update_count + 1

        if args.dataset == 'rotmnist':
            fid_every = 1000
        else:
            fid_every = 2000
        
        if np.mod(gen_update_count, fid_every) == 0:
            # calculate FID using generated samples
            generator.trainable = False
                        
            test_noise = np.random.randn(
                num_classes * num_classes * fid_cal_multiplier, args.latent_dim,
            )
            more_test_noise = np.random.uniform(size=[num_classes * num_classes * fid_cal_multiplier, 2])
            test_noise = np.hstack([test_noise, more_test_noise])

            # calculating samples

            n_batches = test_noise.shape[0]//1000 # drops the last batch if < batch_size

            if args.dataset == 'anhir64' or args.dataset == 'lysto64':
                samples = np.empty((n_batches * 1000, data.shape[1]*2, data.shape[2]*2, data.shape[3]), dtype=np.float32)
            else:
                samples = np.empty((n_batches * 1000, data.shape[1], data.shape[2], data.shape[3]), dtype=np.float32)

            for i in range(n_batches):
                start = i*1000
        
                if start+1000 < test_noise.shape[0]:
                    end = start+1000
                else:
                    end = test_noise.shape[0]
        
                batch_noise = test_noise[start:end]
                batch_labels_fid = test_labels_fid[start:end]
                pred = generator.predict([batch_noise, batch_labels_fid])
                if args.dataset == 'anhir64' or args.dataset == 'lysto64':
                    pred = tf.image.resize(pred, [128, 128])
                samples[start:end] = pred
            
            mu_fake, sigma_fake = calculate_activation_statistics(samples, inception_model, activation_dim, 1000)
                
            fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
            print('\nFID after {}K steps of generator updates: {}'.format(gen_update_count, fid))
            fids.append(fid)

            generator.trainable = True
             

    print('\nTime required for epoch: {}'.format(time() - start_time))

    # Generate samples for visualization and save them:
    generator.trainable = False

    test_noise = np.random.randn(
        num_classes * num_classes * print_multiplier, args.latent_dim,
    )
    # Add two more dimension for noise corresponding to rotation and reflection
    # This is used to implement the correct generator
    more_test_noise = np.random.uniform(size=[num_classes * num_classes * print_multiplier, 2])
    test_noise = np.hstack([test_noise, more_test_noise])

    
    samples = generator.predict([test_noise, test_labels])

    generator.trainable = True
    samples = (samples + 1) / 2.0

    if args.dataset == 'food101':
        n_display = 20
    else:
        n_display = num_classes*print_multiplier

    # For aligning rows with categories:
    for i in range(n_display):
        newrows = np.reshape(
            samples[i * num_classes: i * num_classes + num_classes],
            (data.shape[2] * num_classes, data.shape[2], data.shape[-1]),
        )
        if i == 0:
            rows = newrows
        else:
            rows = np.concatenate((rows, newrows), axis=1)
    rows = np.squeeze(rows)

    plt.imsave('training_pngs/{}/epoch_{:04}.png'
               .format(save_folder, epoch), rows)

    if (epoch + 1) % args.ckpt_freq == 0:
        checkpoint.save(file_prefix=checkpoint_prefix)

# ---------------------------------------------------------------------------
# Save FID results:

fids = np.array(fids)

os.makedirs('result_fids/' + save_folder, exist_ok=True)
np.save('result_fids/{}/fids.npy'.format(save_folder), fids)
