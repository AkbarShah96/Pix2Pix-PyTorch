"""
This is the main training file which governs the training loop
"""

"Imports"
import torch
import torch.nn as nn
from PIL import Image
from models.model import Generator
from data.dataloader import facades
from torchvision import transforms
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

    data_loader = facades(
                            root_dir="C:\\Users\\akbar\\PycharmProjects\\CMP_Facades",
                            transform=[transforms.Resize((256,256), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_loader = DataLoader(data_loader,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=0)
    generator = Generator(args.ngf,
                          args.input_nc,
                          args.output_nc,
                          args.batch_norm)

    discriminator = NLayerDiscriminator(args.ndf,
                                        args.input_nc,
                                        args.output_nc,
                                        args.dropout)

    GAN_loss = torch.nn.MSELoss()
    L1_loss = torch.nn.L1Loss()

    optimizerG = torch.optim.Adam(generator.parameters(),
                                  lr=args['learning_rate'],
                                  weight_decay=0.005,
                                  eps=1e-8)

    optimizerD = torch.optim.Adam(discriminator.parameters(),
                                  lr=args['learning_rate'],
                                  weight_decay=0.005,
                                  eps=1e-8)

    l1_weight = 100

    for epoch in range(args['epoch']):
        l1_loss = 0
        for i, batch in enumerate(data_loader):

            if args['direction'] == "rgb":
                input = batch['semantic'].to(device)
                target = batch['rgb'].to(device)
            elif args['direction'] == "semantic":
                input = batch['rgb'].to(device)
                target = batch['semantic'].to(device)

            "Discriminator Training"
            fake_output = generator.forward(input)
            set_requires_grad(discriminator, True)
            optimizerD.zero_grad()

            "Train on Fake"
            fake_batch = torch.cat((input,fake_output), 1)
            fake_batch.detach()
            pred_fake = discriminator(fake_batch)
            loss_D_fake = GAN_loss(pred_fake, False)

            "Train on Real"
            real_batch = torch.cat((input, target), 1)
            pred_real = discriminator(real_batch)
            loss_D_real = GAN_loss(pred_real, True)

            "Combine Loss and Backprop"
            loss_disc = 0.5*(loss_D_real+loss_D_fake)
            loss_disc.backward()
            optimizerD.step()

            "Generator Training"
            set_requires_grad(discriminator, False)
            optimizerG.zero_grad()

            fake_batch = torch.cat((input, fake_output), 1)
            pred_fake = discriminator(fake_batch)
            loss_G_GAN = GAN_loss(pred_fake, True)
            l1 = L1_loss(fake_output, target)
            total_loss = l1*l1_weight + loss_G_GAN

            l1_loss = l1.item()

            total_loss.backward()
            optimizerG.step()

            print("Epoch:", epoch, "Iter:", i, "L1 loss:", l1_loss, "Disc Loss Real:", loss_D_real, "Disc Loss Fake", loss_D_fake)


if __name__ == '__main__':
    train()             # Calls the main training loop.


