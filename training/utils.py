"""
This file contains functions that help in the training procedure.
"""
"Imports"
import os
import yaml
import torch
import argparse
from easydict import EasyDict


def args_parser():
    parser = argparse.ArgumentParser(description='Pix2Pix GAN Parameters')
    parser.add_argument('--config_path', type=str, default='', help="specify the absolute path to config")
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

def save_yaml(config):
    """
    Saves the parameter in the form of YAML file in the directory where the model is saved
    :param config: (dictionary) contains the parameters
    :return: None
    """
    path = config.SETTINGS.log_path
    if not os.path.exists(path):
        os.makedirs(path)
    with open(os.path.join(path, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)


def load_yaml(path):
    """
    loads a YAML file
    :param path: (string) path to the configuration.yaml file to load
    :return: config file processed into a dictionary by EasyDict
    """
    file = yaml.load(open(path), Loader=yaml.FullLoader)
    config = EasyDict(file)

    return config

