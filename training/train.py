"""
This is the main training file which governs the training loop
"""

"Imports"
import math
import torch
import torch.nn as nn
from PIL import Image
from training.utils import save
from training.loss import GANLossD
from training.loss import GANLossG
from models.model import Generator
from torchvision import transforms
from data.dataloader import dataset
from training.utils import args_parser
from torch.utils.data import DataLoader
from models.model import NLayerDiscriminator


def set_requires_grad(nets, requires_grad):
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

    train_data_loader = dataset(
                            root_dir = "C:\\Users\\akbar\\PycharmProjects\\Pix2Pix-Data",
                            dataset = args['dataset'],
                            mode = 'train',
                            direction = args['direction'],
                            transform=[transforms.Resize((256, 256), Image.BICUBIC),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    val_data_loader = dataset(
                            root_dir="C:\\Users\\akbar\\PycharmProjects\\Pix2Pix-Data",
                            dataset=args['dataset'],
                            mode='val',
                            direction=args['direction'],
                            transform=[transforms.Resize((256, 256), Image.BICUBIC),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_loader = DataLoader(train_data_loader,
                              batch_size=args['batch_size'],
                              shuffle=True,
                              num_workers=0)

    val_loader = DataLoader(val_data_loader,
                              batch_size=args['batch_size'],
                              shuffle=False,
                              num_workers=0)

    "Define Models"
    generator = Generator(args["ngf"],
                          args['input_nc'],
                          args['output_nc'],
                          args['batch_norm'])

    discriminator = NLayerDiscriminator(ndf=args['ndf'],
                                        input_nc=args['input_nc']*2,
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
                             target_real_label_upper=12,
                             target_real_label_lower=8)

    generator_loss = GANLossG('{}'.format(args['loss_mode']),
                             target_fake_label=0.0,
                             target_real_label=1.0)

    discriminator_loss.to(device)
    generator_loss.to(device)


    L1_loss = torch.nn.L1Loss()
    l1_weight = 100
    best_v_loss = math.inf

    for epoch in range(args['epochs']):
        l1_loss = 0
        v_loss = 0
        generator.train()
        discriminator.train()
        for i, batch in enumerate(train_loader):

            input = batch['input'].to(device)
            target = batch['target'].to(device)

            "Discriminator Training"
            fake_output = generator.forward(input)
            set_requires_grad(discriminator, True)
            optimizerD.zero_grad()

            "Train on Fake"
            fake_batch = torch.cat((input, fake_output), 1)
            fake_batch = fake_batch.detach()
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
            l1 = L1_loss(fake_output, target)*l1_weight
            loss_gen = l1 + loss_G_GAN

            loss_gen.backward()
            optimizerG.step()

            torch.cuda.empty_cache()

            print("Epoch:", epoch, "Iter:", i, "L1 loss:", l1.item()/100.0, "Disc Loss Real:", loss_D_real.item(), "Disc Loss Fake", loss_D_fake.item())

            del loss_D_fake, loss_D_real, loss_G_GAN, loss_disc, loss_gen, fake_batch, fake_output, real_batch, pred_real, pred_fake

        generator.eval()
        for i, batch in enumerate(val_loader):
            input = batch['input'].to(device)
            target = batch['target'].to(device)

            "Discriminator Training"
            with torch.no_grad():
                output = generator.forward(input)
            val_loss = L1_loss(output,target)

            v_loss += val_loss

            torch.cuda.empty_cache()

            print("Epoch:", epoch, "Iter:", i, "L1 loss:", val_loss.item()/100.0)

        if v_loss < best_v_loss:
            best_v_loss = v_loss
            save({'epoch': epoch,
                  'state_dict': generator.state_dict()},
                   checkpoint="C:\\Users\\akbar\\PycharmProjects\\Pix2Pix-PyTorch\\checkpoints\\{}".format(args['name']))


if __name__ == '__main__':
    train()             # Calls the main training loop.


