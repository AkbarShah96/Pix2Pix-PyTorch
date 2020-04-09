"""
This file contains functions that help in the training procedure.
"""
"Imports"
import os
import datetime
import torch
import argparse



def args_parser():
    parser = argparse.ArgumentParser(description='Pix2Pix GAN Parameters')
    parser.add_argument('--name', type=str, default='{}'.format(datetime.datetime.now().strftime('%y%m%d-%H%M%S')))
    parser.add_argument('--load', type=str, help="File to test")
    "Model and Data Selection"
    parser.add_argument('--discriminator', type=str, default='n_layers')
    parser.add_argument('--generator', type=str, default='UNet')
    parser.add_argument('--use_dropout', type=bool, default=True)
    parser.add_argument('--batch_norm', type=bool, default=True)
    parser.add_argument('--drop_out', type=int, default=0.5)
    parser.add_argument('--ngf', type=int, default=64, help='Number of Generator Filters')
    parser.add_argument('--ndf', type=int, default=32, help='Number of Discriminator Filters')
    parser.add_argument('--input_nc', type=int, default=3, help='Input Number of Channels RGB = 3, Gray = 1')
    parser.add_argument('--output_nc', type=int, default=3, help='Output Number of Channels RGB = 3, Gray = 1')
    parser.add_argument('--n_layers', type=int, default=3, help='number of layers in discriminator')
    parser.add_argument('--loss_mode', type=str, default='lsgan', help='lsgan, wgangp, or vanilla')
    parser.add_argument('--dataset', type=str, default='facades', help='facades or maps')
    parser.add_argument('--direction', type=str, default='BtoA', help='RGB = A, Semantic = B')
    parser.add_argument('--label_switch', type=str, default=10, help='Noisy label every few iteration to weaken discriminator')

    "Training Hyperparameters"
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=int, default=0.0005)
    parser.add_argument('--train_batch_size', type=int, default=16, help='number of images in a train batch')
    parser.add_argument('--val_batch_size', type=int, default=8, help='number of images in a val batch')
    parser.add_argument('--test_batch_size', type=int, default=1, help='number of images in a test batch')
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--schedular', type=str, default='Step_LR')
    parser.add_argument('--lr_decay', type=int, default=0.8)
    parser.add_argument('--decay_step_size', type=int, default=5)
    args = parser.parse_args()
    return vars(args)

def save(state, checkpoint):
    """
    :param state: State dict
    :param checkpoint: Path to checkpoint
    :return: None, just saves the best dict
    """
    "Path to checkpoint"
    path = os.path.join(checkpoint, 'best.pth.tar')
    "If Path does not exist, create it"
    if not os.path.exists(checkpoint):
        print("Creating Directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    "Save the state dict"
    torch.save(state, path)

def load(net, checkpoint):
    """
    :param net: the model to load
    :param checkpoint: the path to model
    :return: model with loaded weights
    """
    if not os.path.exists(checkpoint):
        raise FileNotFoundError("File does not exist {}".format(checkpoint))
    if torch.cuda.is_available():
        checkpoint = torch.load(checkpoint)
    else:
        checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])

    return checkpoint
