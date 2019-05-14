import torch.nn as nn
import torch


__all__ = ['ShuffleNetV2', 'shufflenet_v2']


def conv_1x1(in_channels, out_channels):
    """1x1 convolution"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)


def dw_conv_3x3(in_channels, out_channels, stride=1):
    """3x3 depthwise convolution and padding 1"""
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)


# Might be to need adjusted if c_prime is different than 2
def channel_shuffle(inp, groups):
    batch_size, num_channels, height, width = inp.data.size()
    channels_per_group = num_channels // groups
    inp = inp.view(batch_size, groups, channels_per_group, height, width)
    inp = torch.transpose(inp, 1, 2).contiguous()
    inp = inp.view(batch_size, -1, height, width)
    return inp


class ShuffleNetV2Unit(nn.Module):
    c_prime = 2

    def __init__(self, in_channels, out_channels, downsample=False, activation=None, norm_layer=None):
        super(ShuffleNetV2Unit, self).__init__()

        if not downsample:
            stride = 1
            in_channels = in_channels // 2
            out_channels -= in_channels
        else:
            stride = 2
            # Number of output channels are distributed between the shortcut connection and the intermediate layers when
            # downsampling
            out_channels = out_channels // self.c_prime

        self.layer = nn.Sequential(
            conv_1x1(in_channels, in_channels),
            norm_layer(in_channels),
            activation,

            dw_conv_3x3(in_channels, in_channels, stride=stride),
            norm_layer(in_channels),

            conv_1x1(in_channels, out_channels),
            norm_layer(out_channels),
            activation,
        )

        if downsample:
            self.shortcut = nn.Sequential(
                dw_conv_3x3(in_channels, in_channels, stride=stride),
                norm_layer(in_channels),

                conv_1x1(in_channels, out_channels),
                norm_layer(out_channels),
                activation,
            )

        self.downsample = downsample
        self.in_channels = in_channels

    def forward(self, x):
        if not self.downsample:
            identity = x[:, :x.size(1) // 2, :, :]
            out = self.layer(x[:, x.size(1) // 2:, :, :])
        else:
            identity = self.shortcut(x)
            out = self.layer(x)

        out = torch.cat((identity, out), 1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    channels = {
        '0.5': [24, 48, 96, 192, 1024],
        '1.0': [24, 116, 232, 464, 1024],
        '1.5': [24, 176, 352, 704, 1024],
        '2.0': [24, 224, 488, 976, 2048],
    }
    layers = [1, 4, 8, 4]

    def __init__(self, num_classes=1000, complexity='1.0', norm_layer=None, activation=None):
        super(ShuffleNetV2, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation is None:
            activation = nn.ReLU(inplace=True)

        # check if complexity configuration exists
        if complexity not in self.channels:
            raise ValueError('Complexity configuration does not exist')

        self.out_channels = self.channels[complexity]

        self.activation = activation
        self.in_channels = self.out_channels[0]

        self.conv1 = nn.Conv2d(3, self.out_channels[0], kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(self.out_channels[0])
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer2 = self._make_layer(self.out_channels[1], blocks=self.layers[1], norm_layer=norm_layer,
                                       activation=activation)
        self.layer3 = self._make_layer(self.out_channels[2], blocks=self.layers[2], norm_layer=norm_layer,
                                       activation=activation)
        self.layer4 = self._make_layer(self.out_channels[3], blocks=self.layers[3], norm_layer=norm_layer,
                                       activation=activation)

        self.conv5 = conv_1x1(self.in_channels, self.out_channels[4])
        self.avg_pool = nn.AvgPool2d(7)
        self.fc = nn.Linear(self.out_channels[4], num_classes)
        self.initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.max_pool(x)

        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

    def _make_layer(self, out_channels, blocks, norm_layer=None, activation=None):
        layers = [ShuffleNetV2Unit(self.in_channels, out_channels, downsample=True, activation=activation,
                                   norm_layer=norm_layer)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(ShuffleNetV2Unit(self.in_channels, out_channels, activation=activation,
                                           norm_layer=norm_layer))
        return nn.Sequential(*layers)

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


def shufflenet_v2(pretrained=False, **kwargs):
    return ShuffleNetV2(**kwargs)
