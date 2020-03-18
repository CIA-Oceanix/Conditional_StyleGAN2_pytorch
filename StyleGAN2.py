from math import floor, log2

import torch
from torch import nn
import torch.nn.functional as F
from torch_optimizer import DiffGrad

from misc import EMA, set_requires_grad
from config import EPS, LATENT_DIM, STYLE_DEPTH, NETWORK_CAPACITY, LEARNING_RATE, CHANNELS, CONDITION_ON_MAPPER

class StyleGAN2(nn.Module):
    def __init__(self, image_size, label_dim, latent_dim=LATENT_DIM, style_depth=STYLE_DEPTH,
                 network_capacity=NETWORK_CAPACITY, steps=1, lr=LEARNING_RATE, channels=CHANNELS,
                 condition_on_mapper=CONDITION_ON_MAPPER):
        super().__init__()
        self.condition_on_mapper = condition_on_mapper
        self.lr = lr
        self.steps = steps
        self.ema_updater = EMA(0.99)

        self.S = StyleVectorizer(latent_dim, label_dim, style_depth, condition_on_mapper=self.condition_on_mapper)
        self.G = Generator(image_size, latent_dim, label_dim, network_capacity, channels=channels,
                           condition_on_mapper=self.condition_on_mapper)
        self.D = Discriminator(image_size, label_dim, network_capacity=network_capacity, channels=channels)

        self.SE = StyleVectorizer(latent_dim, label_dim, style_depth, condition_on_mapper=self.condition_on_mapper)
        self.GE = Generator(image_size, latent_dim, label_dim, network_capacity, channels=channels,
                            condition_on_mapper=self.condition_on_mapper)

        set_requires_grad(self.SE, False)
        set_requires_grad(self.GE, False)

        generator_params = list(self.G.parameters()) + list(self.S.parameters())
        self.G_opt = DiffGrad(generator_params, lr=self.lr, betas=(0.5, 0.9))
        self.D_opt = DiffGrad(self.D.parameters(), lr=self.lr, betas=(0.5, 0.9))

        self._init_weights()
        self.reset_parameter_averaging()

    def _init_weights(self):
        for m in self.modules():
            if type(m) in {nn.Conv2d, nn.Linear}:
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

        for block in self.G.blocks:
            nn.init.zeros_(block.to_noise1.weight)
            nn.init.zeros_(block.to_noise2.weight)
            nn.init.zeros_(block.to_noise1.bias)
            nn.init.zeros_(block.to_noise2.bias)

    def EMA(self):
        def update_moving_average(ma_model, current_model):
            for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
                old_weight, up_weight = ma_params.data, current_params.data
                ma_params.data = self.ema_updater.update_average(old_weight, up_weight)

        update_moving_average(self.SE, self.S)
        update_moving_average(self.GE, self.G)

    def reset_parameter_averaging(self):
        self.SE.load_state_dict(self.S.state_dict())
        self.GE.load_state_dict(self.G.state_dict())

    def forward(self, x):
        return x


class Generator(nn.Module):
    def __init__(self, image_size, latent_dim, label_dim, network_capacity=NETWORK_CAPACITY, channels=CHANNELS,
                 condition_on_mapper=CONDITION_ON_MAPPER):
        super().__init__()
        self.condition_on_mapper = condition_on_mapper
        self.image_size = image_size
        self.latent_dim = latent_dim if self.condition_on_mapper else latent_dim + label_dim
        self.num_layers = int(log2(image_size) - 1)

        init_channels = 4 * network_capacity
        self.initial_block = nn.Parameter(torch.randn((init_channels, 4, 4)))
        filters = [init_channels] + [network_capacity * (2 ** (i + 1)) for i in range(self.num_layers)][::-1]
        in_out_pairs = zip(filters[0:-1], filters[1:])

        self.blocks = nn.ModuleList([])
        for ind, (in_chan, out_chan) in enumerate(in_out_pairs):
            not_first = ind != 0
            not_last = ind != (self.num_layers - 1)

            block = GeneratorBlock(
                self.latent_dim,
                in_chan,
                out_chan,
                upsample=not_first,
                upsample_rgb=not_last,
                channels=channels
            )
            self.blocks.append(block)

    def forward(self, styles, input_noise, labels):
        batch_size = styles.shape[0]
        x = self.initial_block.expand(batch_size, -1, -1, -1)
        styles = styles.transpose(0, 1)

        rgb = None
        for style, block in zip(styles, self.blocks):
            style = style if self.condition_on_mapper else torch.cat((style, labels), 1)
            x, rgb = block(x, rgb, style, input_noise)

        return rgb


class Discriminator(nn.Module):
    def __init__(self, image_size, label_dim, network_capacity=NETWORK_CAPACITY, channels=CHANNELS):
        super().__init__()

        self.label_dim = label_dim
        num_layers = int(log2(image_size) - 1)

        filters = [channels] + [(network_capacity) * (2 ** i) for i in range(num_layers + 1)]
        chan_in_out = list(zip(filters[0:-1], filters[1:]))

        blocks = []
        for ind, (in_chan, out_chan) in enumerate(chan_in_out):
            is_not_last = ind < (len(chan_in_out) - 1)
            block = DiscriminatorBlock(in_chan, out_chan, downsample=is_not_last)
            blocks.append(block)

        self.blocks = nn.Sequential(*blocks)
        self.to_logit = nn.Linear(2 * 2 * filters[-1], label_dim)

    def forward(self, x, labels):
        b, *_ = x.shape
        x = self.blocks(x)
        x = x.reshape(b, -1)
        x = self.to_logit(x)
        x = torch.sum(x * labels, axis=1)
        return x.squeeze()


class GeneratorBlock(nn.Module):
    def __init__(self, latent_dim, input_channels, filters, upsample=True, upsample_rgb=True, channels=CHANNELS):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None

        self.to_style1 = nn.Linear(latent_dim, input_channels)
        self.to_noise1 = nn.Linear(1, filters)
        self.conv1 = Conv2DMod(input_channels, filters, 3)

        self.to_style2 = nn.Linear(latent_dim, filters)
        self.to_noise2 = nn.Linear(1, filters)
        self.conv2 = Conv2DMod(filters, filters, 3)

        self.activation = leaky_relu(0.2)
        self.to_rgb = RGBBlock(latent_dim, filters, upsample_rgb, channels=channels)

    def forward(self, x, prev_rgb, istyle, inoise):
        if self.upsample is not None:
            x = self.upsample(x)

        inoise = inoise[:, :x.shape[2], :x.shape[3], :]
        noise1 = self.to_noise1(inoise).permute((0, 3, 2, 1))
        noise2 = self.to_noise2(inoise).permute((0, 3, 2, 1))

        style1 = self.to_style1(istyle)
        x = self.conv1(x, style1)
        x = self.activation(x + noise1)

        style2 = self.to_style2(istyle)
        x = self.conv2(x, style2)
        x = self.activation(x + noise2)

        rgb = self.to_rgb(x, prev_rgb, istyle)
        return x, rgb


class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, filters, downsample=True):
        super().__init__()
        self.conv_res = nn.Conv2d(input_channels, filters, 1)

        self.net = nn.Sequential(
            nn.Conv2d(input_channels, filters, 3, padding=1),
            leaky_relu(0.2),
            nn.Conv2d(filters, filters, 3, padding=1),
            leaky_relu(0.2)
        )

        self.downsample = nn.Conv2d(filters, filters, 3, padding=1, stride=2) if downsample else None

    def forward(self, x):
        res = self.conv_res(x)
        x = self.net(x)
        x = x + res
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class RGBBlock(nn.Module):
    def __init__(self, latent_dim, input_channel, upsample, channels=CHANNELS):
        super().__init__()
        self.input_channel = input_channel
        self.to_style = nn.Linear(latent_dim, input_channel)
        self.conv = Conv2DMod(input_channel, channels, 1, demod=False)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if upsample else None
        # self.normalize = nn.Sigmoid()

    def forward(self, x, prev_rgb, istyle):
        style = self.to_style(istyle)
        x = self.conv(x, style)

        if prev_rgb is not None:
            x = x + prev_rgb

        if self.upsample is not None:
            x = self.upsample(x)

        return x


class Conv2DMod(nn.Module):
    def __init__(self, in_chan, out_chan, kernel, demod=True, stride=1, dilation=1):
        super().__init__()
        self.filters = out_chan
        self.demod = demod
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        self.weight = nn.Parameter(torch.randn((out_chan, in_chan, kernel, kernel)))
        nn.init.kaiming_normal_(self.weight, a=0, mode='fan_in', nonlinearity='leaky_relu')

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, y):
        b, c, h, w = x.shape

        w1 = y[:, None, :, None, None]
        w2 = self.weight[None, :, :, :, :]
        weights = w2 * (w1 + 1)

        if self.demod:
            d = torch.rsqrt((weights ** 2).sum(dim=(2, 3, 4), keepdims=True) + EPS)
            weights = weights * d

        x = x.reshape(1, -1, h, w)

        _, _, *ws = weights.shape
        weights = weights.reshape(b * self.filters, *ws)

        padding = self._get_same_padding(h, self.kernel, self.dilation, self.stride)
        x = F.conv2d(x, weights, padding=padding, groups=b)

        x = x.reshape(-1, self.filters, h, w)
        return x


class StyleVectorizer(nn.Module):
    def __init__(self, emb, label_dim, depth, condition_on_mapper=CONDITION_ON_MAPPER):
        super().__init__()
        self.condition_on_mapper = condition_on_mapper

        layers = []
        input_shape = (emb + label_dim) if self.condition_on_mapper else emb
        layers.extend([nn.Linear(input_shape, emb), leaky_relu(0.2)])
        for i in range(1, depth):
            layers.extend([nn.Linear(emb, emb), leaky_relu(0.2)])
        self.label_dim = label_dim
        self.net = nn.Sequential(*layers)

    def forward(self, x, labels):
        input_ = torch.cat((x, labels), 1) if self.condition_on_mapper else x
        return self.net(input_)


def leaky_relu(p):
    return nn.LeakyReLU(p, inplace=True)
