from training.train import Pix2Pix
from training.utils import load_yaml
from training.utils import args_parser

"This file calls the main training loop."

if __name__ == '__main__':
    args = args_parser()
    config_path = args['config_path']       # Must give config path through arguments
    config_dict = load_yaml(config_path)    # Loads options from config
    train = Pix2Pix(config_dict)
    train.train_loop()             # Calls the main training loop with options in the config
