import datetime
import os
import numpy as np
import tensorflow as tf
import tensorflow.math as tfm
from scipy import linalg
import warnings

from time import time

from src.utils.data_utils import dataset_lookup, npy_loader, data_generator

import matplotlib
import matplotlib.pyplot as plt

def get_activations(images, model, activation_dim, batch_size=64, verbose=False):
    """Calculates the activations of a given model for all (normalized) images.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, k). The values
                     must lie between [-1, 1]
    -- model       : The given model
    -- activation_dim: dim of the activation
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, d) that contains the
       activations of the given tensor when feeding model with the query tensor.
    """
    n_images = images.shape[0]
    if batch_size > n_images:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = n_images
    n_batches = n_images//batch_size # drops the last batch if < batch_size
    pred_arr = np.empty((n_batches * batch_size,activation_dim))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        
        if start+batch_size < n_images:
            end = start+batch_size
        else:
            end = n_images
        
        batch = images[start:end]
        pred = model.predict(batch)        
        pred_arr[start:end] = pred.reshape(batch.shape[0],-1)
    if verbose:
        print(" done")
    return pred_arr


def calculate_activation_statistics(images, model, activation_dim, batch_size=64, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, k). The values
                     must lie between [-1, 1].
    -- model       : The given model
    -- activation_dim: dim of the activation
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the model
    -- sigma : The covariance matrix of the activations of the model
    """
    act = get_activations(images, model, activation_dim, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
            
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
             for generated samples.
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma1: The covariance matrix over activations of the pool_3 layer for
               generated samples.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, "Training and test mean vectors have different lengths"
    assert sigma1.shape == sigma2.shape, "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = "fid calculation produces singular product; adding %s to diagonal of cov estimates" % eps
        warnings.warn(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def calculate_fid_given_stat_real(images, model, activation_dim, mu_real, sigma_real, batch_size=64, verbose=False):
    """Calculates the FID of generated images with precomputed real statistics"""

    mu_fake, sigma_fake = calculate_activation_statistics(images, model, activation_dim, batch_size, verbose)
    return calculate_frechet_distance(mu_fake, sigma_fake, mu_real, sigma_real, eps=1e-6)
