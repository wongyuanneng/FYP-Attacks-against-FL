#https://github.com/eriklindernoren/PyTorch-GAN

import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

import logging
logger = logging.getLogger('logger')

class Generator(nn.Module):
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.img_shape = img_shape

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(self.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *self.img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self, img_shape):
        super(Discriminator, self).__init__()
        self.img_shape = img_shape

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(self.img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

class PDGAN():
    def __init__(self, n_epochs=400, batch_size=64, lr=0.0002, b1=0.5, b2=0.999, n_cpu=8, latent_dim=100, img_size=28, channels=1, sample_interval=400):
        os.makedirs("images", exist_ok=True)
        self.n_epochs = n_epochs                #number of epochs of training
        self.batch_size = batch_size            #size of the batches
        self.lr = lr                            #adam: learning rate
        self.b1 = b1                            #adam: decay of first order momentum of gradient
        self.b2 = b2                            #adam: decay of first order momentum of gradient
        self.n_cpu = n_cpu                      #number of cpu threads to use during batch generation
        self.latent_dim = latent_dim            #dimensionality of the latent space
        self.img_size = img_size                #size of each image dimension
        self.channels = channels                #number of image channels
        self.sample_interval = sample_interval  #interval betwen image samples

        self.img_shape = (self.channels, self.img_size, self.img_size)

    def build_gan(self):
        # Loss function
        adversarial_loss = torch.nn.BCELoss()

        # Initialize generator and discriminator
        generator = Generator(self.latent_dim,self.img_shape)
        discriminator = Discriminator(self.img_shape)

        if torch.cuda.is_available():
            generator.cuda()
            discriminator.cuda()
            adversarial_loss.cuda()

        # Configure data loader
        os.makedirs("../../data/mnist", exist_ok=True)
        dataloader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "../../data/mnist",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [transforms.Resize(self.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Optimizers
        optimizer_G = torch.optim.Adam(generator.parameters(), lr=self.lr, betas=(self.b1, self.b2))
        optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=self.lr, betas=(self.b1, self.b2))

        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        # ----------
        #  Training
        # ----------

        for epoch in range(self.n_epochs):
            for i, (imgs, _) in enumerate(dataloader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = adversarial_loss(discriminator(gen_imgs), valid)

                g_loss.backward()
                optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = adversarial_loss(discriminator(real_imgs), valid)
                fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                optimizer_D.step()

                logger.error(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(dataloader) + i
                if batches_done % self.sample_interval == 0:
                    save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

pdgan_utility = PDGAN()
pdgan_utility.build_gan()