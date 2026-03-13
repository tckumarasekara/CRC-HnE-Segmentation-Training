import numpy as np
import collections

"""
This file contains parts of code adapted from HistomicsTK
(https://github.com/DigitalSlideArchive/HistomicsTK/), licensed under Apache License 2.0.
"""

def rgb_to_sda(im_rgb):
    """
    From HistomicsTK.
    Convert RGB image to stain density array (SDA) using Beer-Lambert law.

    Parameters:
    - im_rgb (ndarray): Input RGB image

    Returns:
    - im_sda (ndarray): Stain density array
    """

    is_matrix = im_rgb.ndim == 2
    if is_matrix:
        im_rgb = im_rgb.T

    im_rgb = im_rgb.astype(float) + 1
    I_0 = 256

    im_rgb = np.maximum(im_rgb, 1e-10)

    im_sda = -np.log(im_rgb / (1. * I_0)) * 255 / np.log(I_0)
    im_sda = np.maximum(im_sda, 0)

    return im_sda.T if is_matrix else im_sda



def sda_to_rgb(im_sda):
    """
    From HistomicsTK.
    Convert stain density array (SDA) back to RGB image using inverse Beer-Lambert law.

    Parameters:
    - im_sda (ndarray): Stain density array

    Returns:
    - im_rgb (ndarray): Reconstructed RGB image
    """

    is_matrix = im_sda.ndim == 2
    if is_matrix:
        im_sda = im_sda.T

    I_0 = 256

    im_rgb = I_0 ** (1 - im_sda / 255.)
    return (im_rgb.T if is_matrix else im_rgb) - True



def colour_deconvolusion(hne_init, W):
    """
    From HistomicsTK.
    Perform color deconvolution on an image using a given stain matrix.

    Parameters:
    - hne_init (ndarray): Input image
    - W (ndarray): Stain matrix

    Returns:
    - Output (namedtuple): Contains deconvolved stains and related information
    """

    w = np.array(W)

    if w.shape[1] < 3:
        wc = np.zeros((w.shape[0], 3))
        wc[:, :w.shape[1]] = w
        w = wc

    if np.linalg.norm(w[:, 2]) <= 1e-16:
        stain0 = w[:, 0]
        stain1 = w[:, 1]
        stain2 = np.cross(stain0, stain1)
        wc = np.array([stain0, stain1, stain2 / np.linalg.norm(stain2)]).T
    else:
        wc = w

    # normalize stains to unit-norm
    wc = wc / np.sqrt((wc ** 2).sum(0))

    # invert stain matrix
    Q = np.linalg.pinv(wc)

    # transform 3D input image to 2D RGB matrix format
    m_tmp = hne_init if hne_init.ndim == 2 else hne_init.reshape((-1, hne_init.shape[-1])).T
    m = m_tmp[:3]

    # transform input RGB to optical density values and deconvolve,
    # tfm back to RGB
    sda_fwd = rgb_to_sda(m)
    sda_deconv = np.dot(Q, sda_fwd)
    sda_inv = sda_to_rgb(sda_deconv)


    # reshape output
    StainsFloat = sda_inv if len(hne_init.shape) == 2 else sda_inv.T.reshape(hne_init.shape[:-1] + (sda_inv.shape[0],))

    # transform type
    Stains = StainsFloat.clip(0, 255).astype(np.uint8)

    # return
    Unmixed = collections.namedtuple('Unmixed',
                                     ['Stains', 'StainsFloat', 'Wc'])
    Output = Unmixed(Stains, StainsFloat, wc)

    return Output



def colour_deconvolusion_preprocessing_HnE(hne_init):
    """
    Color deconvolution preprocessing for HnE stained images.

    Parameters:
    - hne_init (ndarray): Input HnE stained image

    Returns:
    - hne_deconv (ndarray): Hematoxylin channel after color deconvolution
    """

    # create stain matrix (columns correspond to stains hematoxylin, eosin, null)
    W = np.array([[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.0, 0.0, 0.0]]).T

    # perform standard color deconvolution
    imDeconvolved = colour_deconvolusion(hne_init, W)
    hne_deconv = 1 - imDeconvolved.Stains[:, :, 0]

    return hne_deconv
