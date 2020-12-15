"""
This is the main training file which governs the training loop, please use the main.py to run this file.
"""

"Imports"
import os
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
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.model import NLayerDiscriminator





class Pix2Pix:
    def __init__(self, arguments):
        self.args = arguments
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")                                  # Use GPU if available
        self.log_path = self.args.SETTINGS.log_path
        self.dataset_initialize()
        self.models_initialize()
        self.optimization_initialize()

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))


    def dataset_initialize(self):
        print("-- Preparing Data --")


        self.transform = [transforms.Resize((256, 256), Image.BICUBIC),
                          transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

        self.train_data_loader = dataset(
            root_dir=self.args.DATA.data_path,
            dataset=self.args.DATA.dataset,
            mode='train',
            direction=self.args.DATA.direction,
            transform=self.transform)

        self.val_data_loader = dataset(
            root_dir=self.args.DATA.data_path,
            dataset=self.args.DATA.dataset,
            mode='val',
            direction=self.args.DATA.direction,
            transform=self.transform)

        self.train_loader = DataLoader(self.train_data_loader,
                                  batch_size=self.args.DATA.batch_size,
                                  shuffle=True,
                                  num_workers=self.args.SETTINGS.num_workers,
                                  drop_last=True)

        self.val_loader = DataLoader(self.val_data_loader,
                                batch_size=self.args.DATA.batch_size,
                                shuffle=True,
                                num_workers=self.args.SETTINGS.num_workers,
                                drop_last=True)

        print("-- Dataset DONE --")

    def models_initialize(self):
        print("-- Preparing Models --")

        self.generator = Generator(ngf=self.args.MODEL.ngf,
                              input_nc=self.args.MODEL.input_nc,
                              output_nc =self.args.MODEL.output_nc)

        self.discriminator = NLayerDiscriminator(ndf=self.args.MODEL.ndf,
                                            input_nc=self.args.MODEL.input_nc + self.args.MODEL.output_nc,
                                            n_layers=self.args.MODEL.n_layers)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        print("-- Models DONE --")

    def set_requires_grad(self, nets, requires_grad):
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

    def optimization_initialize(self):

        print("-- Preparing Optimizers and Losses --")
        self.optimizerG = torch.optim.Adam(self.generator.parameters(),
                                      lr=self.args.OPTIMIZATION.gen_learning_rate,
                                      weight_decay=self.args.OPTIMIZATION.weight_decay,
                                      eps=self.args.OPTIMIZATION.eps)

        self.optimizerD = torch.optim.Adam(self.discriminator.parameters(),
                                      lr=self.args.OPTIMIZATION.disc_learning_rate,
                                      weight_decay=self.args.OPTIMIZATION.weight_decay,
                                      eps=self.args.OPTIMIZATION.eps)

        "Initialize Losses"
        self.discriminator_loss = GANLossD(self.args.LOSS.loss_mode)
        self.generator_loss = GANLossG(self.args.LOSS.loss_mode)
        self.discriminator_loss.to(self.device)
        self.generator_loss.to(self.device)
        self.L1_loss = torch.nn.L1Loss()
        self.l1_weight = self.args.LOSS.l1_weight
        self.best_v_loss = math.inf
        self.losses = {}
        self.label_switch = self.args.LOSS.label_switch_noise

        print("-- Optims and Losses DONE --")

    def discriminator_loop(self, input, fake_output, target, i):
        self.set_requires_grad(self.discriminator, True)
        self.optimizerD.zero_grad()

        # Train on Fake

        fake_batch = torch.cat((input, fake_output), 1)
        fake_batch = fake_batch.detach()
        pred_fake = self.discriminator(fake_batch)

        # Adding Noise to fake labels every few iterations
        if i % self.label_switch == 0:
            "Make "
            loss_D_fake = self.discriminator_loss(pred_fake, True)
        else:
            loss_D_fake = self.discriminator_loss(pred_fake, False)

        # Train on Real
        real_batch = torch.cat((input, target), 1)
        pred_real = self.discriminator(real_batch)

        # Adding Noise to real labels every few iterations
        if i % self.label_switch == 0:
            loss_D_real = self.discriminator_loss(pred_real, False)
        else:
            loss_D_real = self.discriminator_loss(pred_real, True)

        # Combine Loss and Backprop
        loss_disc_total = 0.5 * (loss_D_real + loss_D_fake)
        loss_disc_total.backward()
        self.optimizerD.step()

        # For tensorboard and logging
        self.losses["loss_D_fake"] = loss_D_fake.item()
        self.losses["loss_D_real"] = loss_D_real.item()
        self.losses["loss_disc_total"] = loss_disc_total.item()

    def generator_loop(self, input, fake_output, target):
        # Generator Training

        # Turn off discriminator because otherwise its like hitting a moving target
        self.set_requires_grad(self.discriminator, False)
        self.optimizerG.zero_grad()

        # Check discriminator prediction on gens input
        fake_batch = torch.cat((input, fake_output), 1)
        pred_fake = self.discriminator(fake_batch)
        loss_G_GAN = self.generator_loss(pred_fake, True)

        # compute L1 loss
        l1 = self.L1_loss(fake_output, target) * self.l1_weight

        # Add L1 and weighted gen gan loss
        loss_gen = l1 + loss_G_GAN

        loss_gen.backward()
        self.optimizerG.step()

        self.losses["l1"] = l1.item()
        self.losses["loss_G_GAN"] = loss_G_GAN.item()
        self.losses["loss_G_total"] = loss_gen.item()

    def train_loop(self):
        "Runs the main training loop"
        self.step = 0   # defined step for tensorboardX
        for epoch in range(self.args.OPTIMIZATION.epochs):
            self.generator.train()
            self.discriminator.train()

            print("-- Training Epoch {} --".format(epoch))
            for i, batch in enumerate(self.train_loader):

                input = batch['input'].to(self.device)
                target = batch['target'].to(self.device)

                "Discriminator Training"
                fake_output = self.generator(input)
                self.discriminator_loop(input, fake_output, target, i)
                self.generator_loop(input, fake_output, target)
                torch.cuda.empty_cache()

                self.step += 1
                self.log("train", epoch)

                print("Epoch:", epoch,
                      "Iter:", i,
                      "L1 loss:", self.losses["l1"]/self.l1_weight,
                      "Disc Loss Real:", self.losses["loss_D_real"],
                      "Disc Loss Fake", self.losses["loss_D_fake"])

            self.val_loop(epoch)

    def val_loop(self, epoch):
        # Validation after every training epoch, save the best working model
        # You can also save model after every epoch, just remove the if statement below and give a unique save path each time!

        v_loss = 0
        self.generator.eval()

        print("-- Validation Epoch {} --".format(epoch))
        for i, batch in enumerate(self.val_loader):

            input = batch['input'].to(self.device)
            target = batch['target'].to(self.device)

            with torch.no_grad():
                output = self.generator(input)
            val_loss = self.L1_loss(output, target)

            v_loss += val_loss

            torch.cuda.empty_cache()

            print("Epoch:", epoch, "Iter:", i, "L1 loss:", val_loss.item())

        self.v_loss = v_loss.item()/len(self.val_loader)
        self.log("val", epoch)

        print("Validation Loss: ", self.v_loss)

        # Save Best Model
        if self.v_loss < self.best_v_loss:
            print("Saving New Best Model {}".format(epoch))
            self.best_v_loss = v_loss
            save(self.generator.state_dict(), self.log_path)

    def log(self, mode, epoch):
        # You can also edit this function to add images while training

        writer = self.writers[mode]
        if mode == "train":
            for loss, value in self.losses.items():
                writer.add_scalar("{}".format(loss), value, self.step)

        elif mode == "val":
            writer.add_scalar("validation_loss", self.v_loss, epoch)