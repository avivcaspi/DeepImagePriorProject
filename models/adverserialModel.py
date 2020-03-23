from typing import Callable

import torch
import torch.nn as nn

from torch.optim.optimizer import Optimizer
from models.unet import UNet
from models.blocks import DownsampleBlock


class Discriminator(nn.Module):
    def __init__(self, in_size, nd, kd, leaky_slope=0.2):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        self.first_channels = in_size[0]
        self.last_channels = 128

        '''downscaling1 = nn.Sequential(
            nn.Conv2d(self.first_channels, 128, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128)
        )
        downscaling2 = nn.Sequential(
            nn.Conv2d(128, 256, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256)
        )
        downscaling3 = nn.Sequential(
            nn.Conv2d(256, self.last_channels, 4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(self.last_channels)
        )'''
        downsampler = []
        last_depth = self.first_channels
        for num_filters, kernel_size in zip(nd, kd):
            layer = DownsampleBlock(last_depth, num_filters, kernel_size,
                                    leaky_slope=leaky_slope) if num_filters > 0 else None
            downsampler.append(layer)
            last_depth = num_filters
        self.downscaler = nn.Sequential(*downsampler)
        output_size = (in_size[1] // (2 ** len(nd)), in_size[2] // (2 ** len(nd)))
        self.final_layer = nn.Linear(self.last_channels * output_size[0] * output_size[1], 1)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        downscaled = self.downscaler(x)
        reshaped = downscaled.view([x.shape[0], -1])
        y = self.final_layer(reshaped)
        # ========================
        return y


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    device = y_data.device
    loss_fn = nn.BCEWithLogitsLoss()
    data_noise = torch.rand(*y_data.shape) * label_noise - (label_noise / 2)
    generated_noise = torch.rand(*y_data.shape) * label_noise - (label_noise / 2)

    loss_data = loss_fn(y_data, (data_noise + data_label).to(device))
    loss_generated = loss_fn(y_generated, (generated_noise + (1 - data_label)).to(device))
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    device = y_generated.device
    loss_fn = nn.BCEWithLogitsLoss()
    loss = loss_fn(y_generated, torch.full_like(y_generated, data_label, device=device))
    # ========================
    return loss


def train_batch(dsc_model: Discriminator, gen_model: UNet,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                target_img: torch.Tensor, gen_input: torch.Tensor,
                gen_fn: Callable = None, gen_mask: torch.Tensor = None):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()

    real_batch = target_img
    gen_output = gen_model(gen_input)
    generated_batch = gen_output
    if gen_fn is not None:
        generated_batch = gen_fn(generated_batch, gen_mask)
    y_data = dsc_model(real_batch)
    y_generated = dsc_model(generated_batch.detach())

    dsc_loss = dsc_loss_fn(y_data, y_generated)
    dsc_loss.backward()

    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    y_generated = dsc_model(generated_batch)

    gen_loss = gen_loss_fn(y_generated)
    gen_loss.backward()

    gen_optimizer.step()

    # ========================

    return dsc_loss.item(), gen_loss.item(), gen_output

