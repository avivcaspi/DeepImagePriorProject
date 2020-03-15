import torch
import torch.nn as nn


REFLECTION_PADDING = 'reflection'
BILINEAR = 'bilinear'
NEAREST = 'nearest'
UPSAMPLING_METHODS = [BILINEAR, NEAREST]


# Conv -> BN -> LeakyReLU
class ConvBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=nn.BatchNorm2d, leaky_slope=0.2, pad_type=REFLECTION_PADDING, weight_std=0.03):
        nn.Module.__init__(self)

        pad_size = (kernel_size - stride + 1) // 2
        if pad_type == REFLECTION_PADDING:
            self.pad = nn.ReflectionPad2d(pad_size)
        else:
            raise NotImplementedError()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride)
        nn.init.normal_(self.conv.weight, mean=0.0, std=weight_std)

        self.norm = norm_layer(out_channels)
        self.activation = nn.LeakyReLU(leaky_slope)


    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.norm(out)
        out = self.activation(out)
        return out


# (Conv -> BN -> LeakyReLU) -> (StridedConv -> BN -> LeakyReLU)
class DownsampleBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, scale_factor=2, norm_layer=nn.BatchNorm2d, leaky_slope=0.2, pad_type=REFLECTION_PADDING, weight_std=0.03):
        nn.Module.__init__(self)

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, scale_factor, norm_layer, leaky_slope, pad_type, weight_std)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1, norm_layer, leaky_slope, pad_type, weight_std)


    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


# BN -> (Conv -> BN -> LeakyReLU) -> (Conv -> BN -> LeakyReLU) -> Upsample
class UpsampleBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, method, scale_factor=2, norm_layer=nn.BatchNorm2d, leaky_slope=0.2, pad_type=REFLECTION_PADDING, skip_depth=0, weight_std=0.03):
        nn.Module.__init__(self)
        assert method in UPSAMPLING_METHODS

        self.method = method
        self.skip_depth = skip_depth
        in_channels += skip_depth

        self.norm = norm_layer(in_channels)

        self.conv1 = ConvBlock(in_channels, out_channels, kernel_size, 1, norm_layer, leaky_slope, pad_type, weight_std)
        self.conv2 = ConvBlock(out_channels, out_channels, kernel_size, 1, norm_layer, leaky_slope, pad_type, weight_std)

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode=method)


    def forward(self, x, skip_data=None):
        if self.skip_depth > 0:
            assert skip_data is not None
            skip_data = nn.functional.interpolate(skip_data, size=x.shape[2:], mode=self.method)
            x = torch.cat([x, skip_data], dim=1)

        out = self.norm(x)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.upsample(out)
        return out


# (Conv -> BN -> LeakyReLU)
class SkipBlock(nn.Module):


    def __init__(self, in_channels, out_channels, kernel_size, norm_layer=nn.BatchNorm2d, leaky_slope=0.2, pad_type=REFLECTION_PADDING, weight_std=0.03):
        nn.Module.__init__(self)

        self.conv = ConvBlock(in_channels, out_channels, kernel_size, 1, norm_layer, leaky_slope, pad_type, weight_std)


    def forward(self, x):
        out = self.conv(x)
        return out

