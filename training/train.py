"""
This is the main training file which governs the training loop
"""

"Imports"
import torch
import torch.nn as nn
from PIL import Image
from training.loss import GANLossD
from training.loss import GANLossG
from models.model import Generator
from torchvision import transforms
from data.dataloader import dataset
from training.utils import args_parser
from torch.utils.data import DataLoader
from models.model import NLayerDiscriminator


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

def train():

    "Pre-Training"
    args = args_parser()        # Load the arguments

    "Use GPU if available"
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

    data_loader = dataset(
                            root_dir = "C:\\Users\\akbar\\PycharmProjects\\Pix2Pix-Data",
                            dataset = args['dataset'],
                            mode = args['mode'],
                            direction = args['direction'],
                            transform=[transforms.Resize((256, 256), Image.BICUBIC),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = DataLoader(data_loader,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=0)

    "Define Models"
    generator = Generator(args["ngf"],
                          args['input_nc'],
                          args['output_nc'],
                          args['batch_norm'])

    discriminator = NLayerDiscriminator(ndf=args['ndf'],
                                        input_nc=args['input_nc'],
                                        n_layers=args['n_layers'],)

    generator.to(device)
    discriminator.to(device)
    "Initialize Optimizers"

    optimizerG = torch.optim.Adam(generator.parameters(),
                                  lr=args['learning_rate'],
                                  weight_decay=0.005,
                                  eps=1e-8)

    optimizerD = torch.optim.Adam(discriminator.parameters(),
                                  lr=args['learning_rate'],
                                  weight_decay=0.005,
                                  eps=1e-8)

    "Initialize GAN Losses"
    discriminator_loss = GANLossD('{}'.format(args['loss_mode']),
                             target_fake_label=0.0,
                             target_real_label_upper=1.2,
                             target_real_label_lower=0.0)

    generator_loss = GANLossG('{}'.format(args['loss_mode']),
                             target_fake_label=0.0,
                             target_real_label=1.0)

    discriminator_loss.to(device)
    generator_loss.to(device)


    L1_loss = torch.nn.L1Loss()
    l1_weight = 100

    for epoch in range(args['epochs']):
        l1_loss = 0
        for i, batch in enumerate(train_loader):

            input = batch['input'].to(device)
            target = batch['target'].to(device)



            "Discriminator Training"
            fake_output = generator.forward(input)
            set_requires_grad(discriminator, True)
            optimizerD.zero_grad()

            "Train on Fake"
            fake_batch = torch.cat((input,fake_output), 1)
            fake_batch.detach()
            pred_fake = discriminator(fake_batch)
            loss_D_fake = discriminator_loss(pred_fake, False)

            "Train on Real"
            real_batch = torch.cat((input, target), 1)
            pred_real = discriminator(real_batch)
            loss_D_real = discriminator_loss(pred_real, True)

            "Combine Loss and Backprop"
            loss_disc = 0.5*(loss_D_real+loss_D_fake)
            loss_disc.backward()
            optimizerD.step()

            "Generator Training"
            set_requires_grad(discriminator, False)
            optimizerG.zero_grad()

            fake_batch = torch.cat((input, fake_output), 1)
            pred_fake = discriminator(fake_batch)
            loss_G_GAN = generator_loss(pred_fake, True)
            l1 = L1_loss(fake_output, target)
            total_loss = l1*l1_weight + loss_G_GAN

            l1_loss = l1.item()

            total_loss.backward()
            optimizerG.step()

            print("Epoch:", epoch, "Iter:", i, "L1 loss:", l1_loss, "Disc Loss Real:", loss_D_real, "Disc Loss Fake", loss_D_fake)


if __name__ == '__main__':
    train()             # Calls the main training loop.


