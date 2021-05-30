'''
Defines generator and discriminator of the the Generative Adversarial Network (GAN)
for semi-supervised learning using PyTorch library.
It is inspired by Udacity (www.udacity.com) courses and an attempt to rewrite this
implementation in PyTorch:
https://github.com/udacity/deep-learning/blob/master/semi-supervised/semi-supervised_learning_2_solution.ipynb
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.simple import SimpleNet

def recursReverse(Hout, padding, dilation, kernel_size, output_padding, stride):
    return (Hout + 2*padding[0] - dilation[0]*(kernel_size[0]-1) - output_padding[0] - 1)/stride[0] +1

class Generator(nn.Module):
    '''
    The generator network
    '''
    def __init__(self, nz, ngf, alpha, nc, use_gpu=True):
        '''
        :param nz: noise dimension
        :param ngf: generator multiplier for convolution transpose output layers
        :param alpha: negative slope for leaky relu
        :param nc: number of image channels
        :param use_gpu: indication to use the GPU
        '''
        super(Generator, self).__init__()
        self.use_gpu = use_gpu
        
        self.main = nn.Sequential(
            # noise is going into a convolution
            #nz=100, ngf=64. nc=3   64, 100, 1, 1
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(alpha),
            # (ngf * 4) x 4 x 4
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(alpha),
            # (ngf * 2) x 8 x 8
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(alpha),
            # (ngf) x 16 x 16
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # (nc) x 32 x 32
        )

    def forward(self, inputs):
        '''
        :param inputs: we expect noise as input for generator network
        '''
        if isinstance(inputs.data, torch.cuda.FloatTensor) and self.use_gpu:
            out = nn.parallel.data_parallel(self.main, inputs, range(1))
        else:
            out = self.main(inputs)
        return out


class _ganLogits(nn.Module):
    '''
    Layer of the GAN logits of the discriminator
    The layer gets class logits as inputs and calculates GAN logits to
    differentiate real and fake images in a numerical stable way
    '''
    def __init__(self, num_classes):
        '''
        :param num_classes: Number of real data classes (10 for SVHN)
        '''
        super(_ganLogits, self).__init__()
        self.num_classes = num_classes

    def forward(self, class_logits):
        '''
        :param class_logits: Unscaled log probabilities of house numbers
        '''

        # Set gan_logits such that P(input is real | input) = sigmoid(gan_logits).
        # Keep in mind that class_logits gives you the probability distribution over all the real
        # classes and the fake class. You need to work out how to transform this multiclass softmax
        # distribution into a binary real-vs-fake decision that can be described with a sigmoid.
        # Numerical stability is very important.
        # You'll probably need to use this numerical stability trick:
        # log sum_i exp a_i = m + log sum_i exp(a_i - m).
        # This is numerically stable when m = max_i a_i.
        # (It helps to think about what goes wrong when...
        #   1. One value of a_i is very large
        #   2. All the values of a_i are very negative
        # This trick and this value of m fix both those cases, but the naive implementation and
        # other values of m encounter various problems)
        real_class_logits, fake_class_logits = torch.split(class_logits, self.num_classes, dim=1)
        fake_class_logits = torch.squeeze(fake_class_logits)

        max_val, _ = torch.max(real_class_logits, 1, keepdim=True)
        stable_class_logits = real_class_logits - max_val
        max_val = torch.squeeze(max_val)
        gan_logits = torch.log(torch.sum(torch.exp(stable_class_logits), 1)) + max_val - fake_class_logits

        return gan_logits


class Discriminator(SimpleNet):
    '''
    The discriminator network
    '''
    def __init__(self, global_model, ndf=256, num_classes=10, alpha=.2, nc=1, drop_rate=.5):
        '''
        :param ndf: multiplier for convolution output layers
        :param alpha: negative slope for leaky relu
        :param nc: number of image channels
        :param drop_rate: rate for dropout layers
        :param num_classes: number of output classes (10 for SVHN)
        '''
        super(Discriminator, self).__init__(num_classes)

        self.main = global_model
        #class_logits
        self.main.fc = nn.Linear(in_features=(ndf * 2) * 1 * 1, out_features=num_classes + 1)

        self.gan_logits = _ganLogits(num_classes)

        self.softmax = nn.Softmax(dim=0)

    def forward(self, inputs):
        '''
        :param inputs: we expect real or fake images as an input for discriminator network
        '''

        class_logits = self.main(inputs)

        #features = self.features(out)
        #features = features.squeeze()

        #class_logits = self.class_logits(features)

        #gan_logits = self.gan_logits(class_logits)
        gan_logits = self.gan_logits(class_logits)

        out = self.softmax(class_logits)

        return out, class_logits, gan_logits

