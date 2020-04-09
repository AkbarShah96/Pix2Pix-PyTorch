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

def test():

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


    test_loader = DataLoader(test_data_loader,
                              batch_size=args['test_batch_size'],
                              shuffle=False,
                              num_workers=0)

    "Define Models"
    generator = Generator(args["ngf"],
                          args['input_nc'],
                          args['output_nc'],
                          args['batch_norm'])
    "Load weights"
    checkpoint = "C:\\Users\\akbar\\PycharmProjects\\Pix2Pix-PyTorch\\checkpoints\\{}".format(args['load'])
    checkpoint = os.path.join(checkpoint, "best.pth.tar")
    load(generator, checkpoint)
    generator.to(device)



    L1_loss = torch.nn.L1Loss()

    for epoch in range(args['epochs']):
        t_loss = 0

        generator.eval()
        for i, batch in enumerate(test_loader):
            input = batch['input'].to(device)
            target = batch['target'].to(device)

            "Discriminator Training"
            with torch.no_grad():
                output = generator.forward(input)
            val_loss = L1_loss(output,target)

            t_loss += val_loss

            torch.cuda.empty_cache()

            "Display Images"
            input = input.cpu().numpy()
            input = input.squeeze().transpose((1, 2, 0))

            output = output.cpu().numpy()
            output = output.squeeze().transpose((1,2,0))


            fig = plt.figure()

            ax1 = fig.add_subplot(1, 2, 1)
            ax1.set_title('input')
            ax1.imshow(input)

            ax2 = fig.add_subplot(1, 2, 2)
            ax2.set_title('output')
            ax2.imshow(output)

            plt.pause(0.001)

            print("Epoch:", epoch, "Iter:", i, "L1 loss:", val_loss.item())

if __name__ == '__main__':
    test()             # Calls the main training loop.


