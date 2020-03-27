"""
This is the main training file which governs the training loop
"""

"Imports"
import torch
import torch.nn as nn
from PIL import Image
from training.utils import load
from models.model import Generator
from torchvision import transforms
from data.dataloader import dataset
from training.utils import args_parser
from torch.utils.data import DataLoader


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

    test_data_loader = dataset(
                            root_dir = "C:\\Users\\akbar\\PycharmProjects\\Pix2Pix-Data",
                            dataset = args['dataset'],
                            mode = 'test',
                            direction = args['direction'],
                            transform=[transforms.Resize((256, 256), Image.BICUBIC),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


    val_loader = DataLoader(test_data_loader,
                              batch_size=args['batch_size'],
                              shuffle=False,
                              num_workers=0)

    "Define Models"
    generator = Generator(args["ngf"],
                          args['input_nc'],
                          args['output_nc'],
                          args['batch_norm'])

    checkpoint = "C:\\Users\\akbar\\PycharmProjects\\Pix2Pix-PyTorch\\checkpoints\\{}".format(args['load'])
    load(generator, checkpoint)
    generator.to(device)



    L1_loss = torch.nn.L1Loss()

    for epoch in range(args['epochs']):
        t_loss = 0

        generator.eval()
        for i, batch in enumerate(val_loader):
            input = batch['input'].to(device)
            target = batch['target'].to(device)

            "Discriminator Training"
            with torch.no_grad():
                output = generator.forward(input)
            val_loss = L1_loss(output,target)

            t_loss += val_loss

            torch.cuda.empty_cache()

            print("Epoch:", epoch, "Iter:", i, "L1 loss:", val_loss.item()/100.0)

if __name__ == '__main__':
    train()             # Calls the main training loop.


