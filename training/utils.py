"""
This file contains functions that help in the training procedure.
"""
"Imports"
import argparse



def args_parser():
    parser = argparse.ArgumentParser(description='Pix2Pix GAN Parameters')
    "Model Selection"
    parser.add_argument('--discriminator', type=str, default='n_layers')
    parser.add_argument('--generator', type=str, default='UNet')
    parser.add_argument('--use_dropout', type=bool, default=True)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--ngf', type=int, default=64, help='Number of Generator Filters')
    parser.add_argument('--ndf', type=int, default=64, help='Number of Discriminator Filters')
    parser.add_argument('--input_nc', type=int, default=3, help='Input Number of Channels RGB = 3, Gray = 1')
    parser.add_argument('--output_nc', type=int, default=3, help='Output Number of Channels RGB = 3, Gray = 1')
    parser.add_argument('--loss_mode', type=str, default='lsgan', help='lsgan, WgGan, or Vanilla')
    parser.add_argument('--direction', type=str, default='rgb', help='What is output? rgb or semantic?')

    "Training Hyperparameters"
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=int, default=0.0001)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--schedular', type=str, default='Step_LR')
    parser.add_argument('--lr_decay', type=int, default=0.95)
    parser.add_argument('--decay_step_size', type=int, default=5)

    return parser

