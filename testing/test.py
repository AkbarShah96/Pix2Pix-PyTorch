"""
This is the main training file which governs the training loop
"""

"Imports"
import os
import torch
from PIL import Image

from training.utils import load
from models.model import Generator
from torchvision import transforms
from data.dataloader import dataset
from matplotlib import pyplot as plt
from training.utils import load_yaml
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

def test(args):
    device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
    transform = [transforms.Resize((256, 256), Image.BICUBIC),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

    test_data_loader = dataset(
                            root_dir = args.DATA.data_path,
                            dataset = args.DATA.dataset,
                            mode = 'test',
                            direction = args.DATA.direction,
                            transform=transform)


    test_loader = DataLoader(test_data_loader,
                              batch_size=1,
                              shuffle=True,
                              num_workers=4)

    "Define Models"
    generator = Generator(args.MODEL.ngf,
                          args.MODEL.input_nc,
                          args.MODEL.output_nc)
    "Load weights"
    checkpoint = args.EVALUATION.evaluation_path
    checkpoint = os.path.join(checkpoint, "best.pth.tar")
    load(generator, checkpoint)
    generator.to(device)
    L1_loss = torch.nn.L1Loss()


    t_loss = 0
    generator.eval()

    for i, batch in enumerate(test_loader):
        input = batch['input'].to(device)
        target = batch['target'].to(device)

        "Discriminator Training"
        with torch.no_grad():
            output = generator.forward(input)
        test_loss = L1_loss(output,target)

        t_loss += test_loss



        if args.EVALUATION.plot:
            "Display Images"
            input = input.detach().cpu().numpy()
            input = input.squeeze().transpose((1, 2, 0))

            output = output.detach().cpu().numpy()
            output = output.squeeze().transpose((1, 2, 0))

            fig = plt.figure()
            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title('input')
            ax1.imshow(input)

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title('output')
            ax2.imshow(output)
            plt.pause(1)

        print("Iter:", i, "L1 loss:", test_loss.item())
    print("Final Test Loss: ", t_loss.item()/len(test_loader))

if __name__ == '__main__':
    args = args_parser()
    config_path = args['config_path']       # Must give config path through arguments
    config_dict = load_yaml(config_path)    # Loads options from config
    test(config_dict)                       # Calls the main test loop.


