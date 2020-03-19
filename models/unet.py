import torch
import torch.nn as nn
from .blocks import *


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, nu, nd, ns, ku, kd, ks, up_method=BILINEAR, leaky_slope=0.2):
        nn.Module.__init__(self)

        self.downs = []
        self.ups = []
        self.skips = []

        # Create downsample part ('encoder')
        last_depth = in_channels
        for num_filters, kernel_size in zip(nd, kd):
            layer = DownsampleBlock(last_depth, num_filters, kernel_size, leaky_slope=leaky_slope) if num_filters > 0 else None
            self.downs.append(layer)
            last_depth = num_filters
        self.downs = nn.ModuleList(self.downs)

        # Create upsample part ('decoder') in reverse order
        for i, (num_filters, kernel_size) in reversed(list(enumerate(zip(nu, ku)))):
            skip_depth = ns[i]

            layer = UpsampleBlock(last_depth, num_filters, kernel_size, up_method, skip_depth=skip_depth, leaky_slope=leaky_slope) if num_filters > 0 else None
            self.ups.append(layer)
            last_depth = num_filters

        self.ups = list(reversed(self.ups))
        self.ups = nn.ModuleList(self.ups)

        # Create skip connection blocks
        last_depth = in_channels
        for i, (num_filters, kernel_size) in enumerate(zip(ns, ks)):
            layer = SkipBlock(last_depth, num_filters, kernel_size, leaky_slope=leaky_slope) if num_filters > 0 else None
            self.skips.append(layer)
            last_depth = nd[i]
        self.skips = nn.ModuleList(self.skips)

        final_conv = nn.Conv2d(nu[0], out_channels, 1)
        self.final = nn.Sequential(final_conv, nn.Sigmoid())


    def forward(self, x):
        main_data = x.clone()
        skips_data = []

        for i, down_layer in enumerate(self.downs):
            curr_skips_data = main_data.clone() if self.skips[i] else None
            skips_data.append(curr_skips_data)
                
            main_data = down_layer(main_data)

        for i, skip_layer in enumerate(self.skips):
            if skip_layer:
                skips_data[i] = skip_layer(skips_data[i])

        for i, up_layer in reversed(list(enumerate(self.ups))):
            main_data = up_layer(main_data, skips_data[i])

        out = self.final(main_data)

        return out
