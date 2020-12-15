"""
This file contains the Generator and Discriminator Models.
I translated these models from https://www.tensorflow.org/tutorials/generative/pix2pix
"""
import torch
import torch.nn as nn

class down_sample(nn.Module):
    def __init__(self,input_nc, ngf, drop_out = False):
        """
        This class down samples through the network
        :param input_nc: int
        :param ngf: int
        :param batch_norm: bool
        """
        super(down_sample, self).__init__()

        if drop_out:
            self.sequence = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.Dropout(0.2),
                nn.LeakyReLU()
            )
        else:
            self.sequence = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.sequence(x)

class up_sample(nn.Module):
    def __init__(self, input_nc, ngf, drop_out = False):
        """
        This class up samples through the network
        :param input_nc: int
        :param ngf:  int
        :param drop_out: bool
        """
        super(up_sample, self).__init__()

        if drop_out:
            self.sequence = nn.Sequential(
                nn.ConvTranspose2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.Dropout(0.5),
                nn.LeakyReLU()
            )
        else:
            self.sequence = nn.Sequential(
                nn.ConvTranspose2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.sequence(x)

class Generator(nn.Module):
    def __init__(self, ngf,input_nc, output_nc):
        """
        This is a Generator architecture with skip connections.
        :param ngf: number of filters
        :param input_nc: number of input channels
        :param output_nc: number of output channels
        :param batch_norm: use batch norm
        """
        super(Generator, self).__init__()
        self.down_stack = nn.Sequential(
                        down_sample(input_nc, ngf),
                        down_sample(ngf, ngf * 2),
                        down_sample(ngf * 2, ngf * 4),
                        down_sample(ngf * 4, ngf * 8),
                        down_sample(ngf * 8, ngf * 8),
                        down_sample(ngf * 8, ngf * 8),
                        down_sample(ngf * 8, ngf * 8),
                        down_sample(ngf * 8, ngf * 8)
        )

        self.up_stack = nn.Sequential(
                        up_sample(ngf * 8, ngf * 8, drop_out= True),
                        up_sample(ngf * 16, ngf * 8, drop_out= True),
                        up_sample(ngf * 16, ngf * 8, drop_out= True),
                        up_sample(ngf * 16, ngf * 8, drop_out= True),
                        up_sample(ngf * 16, ngf * 4),
                        up_sample(ngf * 8, ngf * 2),
                        up_sample(ngf * 4, ngf),
                        nn.Upsample(scale_factor=2, mode='bilinear'),
        )
        self.last = nn.Sequential(
            nn.ConvTranspose2d(ngf*2, output_nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        "Down Sample Through Model"
        skips = []
        for down in self.down_stack:
            x = down(x)
            skips.append(x)     # Store each x to form skip connection to upsample

        skips = reversed(skips[:-1])

        "Upsampling Through Model"
        for up, skip in zip(self.up_stack, skips):
            x = up(x)
            x = torch.cat((x, skip), dim=1)
        x = self.last(x)
        return x

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator, adopted from Junyaz
       Added comments and cleaned code.
       The result of a PatchGAN is a feature map which tells us if each patch is True or False
    """

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        "Start a sequence with conv layer + leakyRelu"
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, True)]
        "Initialise constants that will control the number of filters in each layer as we progress"
        nf_mult = 1
        nf_mult_prev = 1

        "Depending on n_layers we will build a discriminator"
        for n in range(1, n_layers):
            "Update Values to have correct channel sizes"
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)    #limit max channels size to ndf * 8.

            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]


        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)

        "Second Last Layer"
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        "Final layer"
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]  # output 1 channel prediction map

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """forward through discriminator"""
        return self.model(input)

