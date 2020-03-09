"""
This file contains the Generator and Discriminator Models.
I translated these models from https://www.tensorflow.org/tutorials/generative/pix2pix
"""
import torch
import torch.nn as nn

class down_sample(nn.Module):
    def __init__(self,input_nc, ngf, batch_norm):
        """
        This class down samples through the network
        :param input_nc: int
        :param ngf: int
        :param batch_norm: bool
        """
        super(down_sample, self).__init__()

        if batch_norm:
            self.sequence = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding='same', bias=False),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU
            )
        else:
            self.sequence = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding='same', bias=False),
                nn.LeakyReLU
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
                nn.ConvTranspose2d(input_nc, ngf, kernel_size=4, stride=2, padding='same', bias=False),
                nn.BatchNorm2d(ngf),
                nn.Dropout(0.5),
                nn.LeakyReLU
            )
        else:
            self.sequence = nn.Sequential(
                nn.ConvTranspose2d(input_nc, ngf, kernel_size=4, stride=2, padding='same', bias=False),
                nn.BatchNorm2d(ngf),
                nn.LeakyReLU
            )


class Generator(nn.Module):
    def __init__(self, ngf, ndf, input_nc, output_nc, batch_norm):
        super(Generator, self).__init__()

        self.down_stack = nn.Sequential(
                        down_sample(input_nc, ngf, batch_norm),
                        down_sample(ngf, ngf * 2, batch_norm),
                        down_sample(ngf * 2, ngf * 4, batch_norm),
                        down_sample(ngf * 4, ngf * 8, batch_norm),
                        down_sample(ngf * 8, ngf * 8, batch_norm),
                        down_sample(ngf * 8, ngf * 8, batch_norm),
                        down_sample(ngf * 8, ngf * 8, batch_norm),
                        down_sample(ngf * 8, ngf * 8, batch_norm))

        self.up_stack = nn.Sequential(
                        up_sample(ngf * 8, ngf * 8, drop_out= True),
                        up_sample(ngf * 8, ngf * 8, drop_out= True),
                        up_sample(ngf * 8, ngf * 8, drop_out= True),
                        up_sample(ngf * 8, ngf * 8),
                        up_sample(ngf * 8, ngf * 4),
                        up_sample(ngf * 4, ngf * 2),
                        up_sample(ngf * 2, ngf))

        self.last = nn.Sequential(
            nn.ConvTranspose2d(ngf, output_nc, stride=2, padding='same'),
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
            x = torch.cat(x,skip)

        return self.last(x)



