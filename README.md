# Pix2Pix-PyTorch
 
This repo contains a simplified implementation of pix2pix which is one of my favourite GANs. 

Paper: [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf) 

Code: [Junyaz's PyTorch Implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)

Status: *In Progress*

## Introduction


## Markovian Discriminator (PatchGAN)
A PatchGAN is a discriminator architecture which penalizes structure within a certain patch and classifies if each patch is real or fake. The PatchGAN architecture consists of fully convolutional layers, hence the output is a feature map consisting of classification (real or fake) of the patch. The values in the feature maps are like probabilities of being real or fake. The size of the patches can be controlled with the number of layers and strided convolutions in the discriminator. Intuitively, less layers result in large feature maps hence large patches and more layers result in smaller patches. The smallest patch size is 1x1 and the largest patch size is the image size for square images. The advantage of such a discriminator is that it models the image as a Markov random field, where the pixels are independent when they are separated by more than a patch diameter. Therefore, great emphasis is put on style and texture of the image. 

<p align="center">
  <img src="readme/PatchGAN.png" width="600px"/>
</p>


## Loss Functions 
L2 is a popular loss for image processing tasks but it produces blurry results and does not correlate well with image quality as humans perceive it. In comparison, the L1 loss has improved performance but it's still not optimal. Both of these losses do not encourage high-frequency crispness but they do capture low frequencies well. Therefore, pix2pix uses L1 for correctness at low frequencies and restricts the PatchGAN for structure at high frequencies. For modelling high frequencies, the focus is restricted to structure in local image patches. 

The loss that the PatchGAN uses is from [LSGAN](https://arxiv.org/pdf/1611.04076.pdf). The loss function is able to move the fake samples closer to the decision boundary between real and fake samples. LSGAN moves the fake samples closer to the real samples since the loss function penalizes samples that lie too far from the decision boundary on the correct side. If the samples are on the correct side of the decision boundary but still far from the real data, they tend to suffer from vanishing gradient problem. LSGAN remedies this problem since it forces far samples to be closer. 

<p align="center">
  <img src="readme/LSGAN.png" width="300px"/>
</p>

In the equation above, a is the fake sample and b is the real sample, and c is the value that the generator wants the discriminator to believe for fake data.  

## Training Procedure


