import torch
import numpy as np


def uniform_noise_gen(depth, height, width, std=1./10):
    noise = torch.empty(depth, height, width).uniform_()
    noise *= std
    return noise


def normal_noise_gen(depth, height, width, std=1./10):
    noise = torch.empty(depth, height, width).normal_()
    noise *= std
    return noise


def meshgrid_gen(height, width):
    X, Y = np.meshgrid(np.arange(0, width)/float(width - 1), np.arange(0, height)/float(height - 1))
    meshgrid = np.concatenate([X[None, :], Y[None, :]])
    return torch.from_numpy(meshgrid)


def get_noise(depth, height, width, method, noise_type='uniform', std=1./10):
    """
    create noise tensor with the shape as requested according to type and method given
    :param depth: the depth of the noise tensor, if method is meshgrid - depth will be 2
    :param height: the height of the noise tensor
    :param width: the width of the noise tensor
    :param method: can be 'noise' for normal or uniform noise, or 'meshgrid' for meshgrid
    :param noise_type: can be 'uniform' or 'normal' for uniform or normal noise type
    :param std: standard deviation of the noise
    :return:
    """
    if method == 'noise':
        if noise_type == 'uniform':
            noise = uniform_noise_gen(depth, height, width, std)
        elif noise_type == 'normal':
            noise = normal_noise_gen(depth, height, width, std)
        else:
            assert False
    elif method == 'meshgrid':
        noise = meshgrid_gen(height, width)
    else:
        assert False
        
    return noise


def add_noise_to_tensor(img, noise_type='uniform', std=1./10):
    assert img.dim() == 3
    noise = get_noise(*img.shape, method='noise', noise_type=noise_type, std=std)
    noisy_img = np.clip(img + noise, 0, 1)
    return noisy_img, noise


