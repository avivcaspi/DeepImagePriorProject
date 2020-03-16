import math
import torch

import numpy as np
import matplotlib.pyplot as plt

import PIL
from PIL import Image


def tensors_as_images(tensors, nrows=1, figsize=(8, 8), titles=[],
                      wspace=0.1, hspace=0.2, cmap=None):
    """
    Plots a sequence of pytorch tensors as images.

    :param tensors: A sequence of pytorch tensors, should have shape CxWxH
    """
    assert nrows > 0

    num_tensors = len(tensors)

    ncols = math.ceil(num_tensors / nrows)

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                             gridspec_kw=dict(wspace=wspace, hspace=hspace),
                             subplot_kw=dict(yticks=[], xticks=[]))
    axes_flat = axes.reshape(-1)

    # Plot each tensor
    for i in range(num_tensors):
        ax = axes_flat[i]

        image_tensor = tensors[i]
        assert image_tensor.dim() == 3  # Make sure shape is CxWxH
        image = image_tensor.cpu().numpy()
        image = image.transpose(1, 2, 0)
        image = image.squeeze()  # remove singleton dimensions if any exist


        # Scale to range 0..1
        min, max = np.min(image), np.max(image)
        image = (image-min) / (max-min)
        if len(image.shape) == 2:
            image = 1-image

        ax.imshow(image, cmap=cmap)

        if len(titles) > i and titles[i] is not None:
            ax.set_title(titles[i])

    # If there are more axes than tensors, remove their frames
    for j in range(num_tensors, len(axes_flat)):
        axes_flat[j].axis('off')
    plt.show()
    return fig, axes


def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img


def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.
    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np


def pil_to_np(img_PIL):
    """Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2, 0, 1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    """Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def crop_image(img, d=32):
    """Make dimensions divisible by `d`"""

    new_size = (img.size[0] - img.size[0] % d,
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2),
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped


def get_bernoulli_mask(dims, p):
    assert isinstance(dims, tuple)
    assert len(dims) == 3
    assert 0 <= p <= 1
    mask = np.random.choice([0, 1], size=dims, p=[p, 1-p])
    return mask


def lanczos2_kernel(factor, phase=0):
    support = 2
    kernel_size = 4 * factor + 1
    center = (kernel_size + 1.0) / 2.0
    
    kernel = np.zeros([kernel_size, kernel_size])
    
    for i in range(1, kernel_size+1):
        for j in range(1, kernel_size+1):
            
            di = abs(i - center + phase) / factor
            dj = abs(j - center + phase) / factor
            
            value = 1
            for offset in [di, dj]:
                if abs(offset) < 1e-9:   # if offset == 0
                    continue
                value *= support
                value *= np.sin(np.pi * offset) * np.sin(np.pi * offset / support)
                value /= (np.pi * offset) ** 2
                
            kernel[i-1][j-1] = value
            
    return kernel / kernel.sum()


def lanczos_downsample(tensor, factor, pad_type=torch.nn.ReplicationPad2d, device='cpu'):
    kernel_np = lanczos2_kernel(factor, phase=0.5)
    kernel_tr = torch.from_numpy(kernel_np)
    
    depth = tensor.shape[1]
    conv = torch.nn.Conv2d(depth, depth, kernel_size=kernel_tr.shape, stride=factor, padding=0).to(device)
    
    conv.weight.data[:] = 0
    conv.bias.data[:] = 0
    
    for i in range(depth):
        conv.weight.data[i, i] = kernel_tr
        
    pad_size = (kernel_np.shape[0] - 1) // 2
    pad = pad_type(pad_size).to(device)
        
    return conv(pad(tensor))

