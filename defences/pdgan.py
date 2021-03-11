#https://towardsdatascience.com/understanding-acgans-with-code-pytorch-2de35e05d3e4
#https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/acgan/acgan.py 

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
import copy

from helper import Helper
from models.resnet import ResNet

import logging
logger = logging.getLogger('logger')

class Generator(nn.Module):
    #generator model
    def __init__(self,in_channels):
        super(Generator,self).__init__()
        self.fc1=nn.Linear(in_channels,384)

        self.t1=nn.Sequential(
            nn.ConvTranspose2d(in_channels=384,out_channels=192,kernel_size=(4,4),stride=1,padding=0),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.t2=nn.Sequential(
            nn.ConvTranspose2d(in_channels=192,out_channels=96,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU()
        )
        self.t3=nn.Sequential(
            nn.ConvTranspose2d(in_channels=96,out_channels=48,kernel_size=(4,4),stride=2,padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU()
        )
        self.t4=nn.Sequential(
            nn.ConvTranspose2d(in_channels=48,out_channels=3,kernel_size=(4,4),stride=2,padding=1),
            nn.Tanh()
        )
    
    def forward(self,x):
    	x=x.view(-1,110)
    	x=self.fc1(x)
    	x=x.view(-1,384,1,1)
    	x=self.t1(x)
    	x=self.t2(x)
    	x=self.t3(x)
    	x=self.t4(x)
    	return x #output of generator


class Discriminator(nn.Module):
    def __init__(self, global_model):
        super(Discriminator, self).__init__()

        self.model = global_model
        self.model.fc = nn.Linear(512, 1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1)
        validity = self.sig(x)
        
        return validity

class PDGAN():
    def __init__(self):
        os.makedirs("images", exist_ok=True)

        self.generator = None
        self.discriminator = None
    
    def compute_acc(self, preds, labels):
        correct = 0
        preds_ = preds.data.max(1)[1]
        correct = preds_.eq(labels.data).cpu().sum()
        acc = float(correct) / float(len(labels.data)) * 100.0
        return acc

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def run_defence(self, hlpr, aux, global_model, participant_updates, round_no, epochs=100, lr=0.0002, batch_size=64):
        if hlpr.params.fl_pdgan == 0:
            return participant_updates
            
        benign_update_list = []

        #set up aux data
        aux_loader = aux

        # # Initialize generator and discriminator if not exist
        if self.generator == None:
            self.generator = Generator(110).cuda()
            self.generator.apply(self.weights_init)

        x_fake = self.generate_fake(batch_size)
        
        # update discriminator
        weight_accumulator = hlpr.task.get_empty_accumulator()
        updated_global_model = copy.deepcopy(global_model)
        for updates in participant_updates:
            hlpr.task.accumulate_weights(weight_accumulator, updates)
        hlpr.task.update_global_model(weight_accumulator, updated_global_model)
        self.discriminator = Discriminator(updated_global_model).cuda()

        self.train(aux_loader, batch_size, epochs, lr)
       
        if round_no > hlpr.params.fl_pdgan:
            ##Perform accuracy auditing
            for k in range(len(participant_updates)):
                #initialise participant classification model
                updated_global_model_k = copy.deepcopy(global_model)
                weight_accumulator = hlpr.task.get_empty_accumulator()
                hlpr.task.accumulate_weights(weight_accumulator, participant_updates[k])
                hlpr.task.update_global_model(weight_accumulator, updated_global_model_k)
                
                hlpr.task.reset_metrics()
                with torch.no_grad():
                    #Assign labels for x_fake based on L
                    for i, data in enumerate(aux_loader):
                        batch = hlpr.task.get_batch(i, data)
                        outputs = updated_global_model_k(batch.inputs)
                        hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
                #Calculate accuracy a of each participant classification model on x_fake
                metric = hlpr.task.metrics[0].get_main_metric_value()
                logger.warning(f"x_fake for participant {k}. Epoch: {round_no}. Accuracy: {metric}")

                if (metric >= accuracy_threshold):
                    benign_update_list.append(participant_updates[k])
            return benign_update_list
        else:
            return participant_updates
    
    def generate_fake(self, batch_size):
        noise_ = np.random.normal(0, 1, (batch_size, 110))        #generating noise by random sampling from a normal distribution
        label = np.random.randint(0,10,batch_size)                #generating labels for the entire batch
        
        noise = ((torch.from_numpy(noise_)).float())
        noise = noise.cuda()                                      #converting to tensors in order to work with pytorch

        label = ((torch.from_numpy(label)).long())
        label = label.cuda()                                      #converting to tensors in order to work with pytorch
        
        return self.generator(noise)

    def train(self, dataloader, batch_size, epochs, lr):
        real_label = torch.FloatTensor(batch_size).cuda()
        real_label.fill_(1)

        fake_label = torch.FloatTensor(batch_size).cuda()
        fake_label.fill_(0)
        
        # eval_noise = torch.FloatTensor(batch_size, 110, 1, 1).normal_(0, 1)
        # eval_noise_ = np.random.normal(0, 1, (batch_size, 110))
        # eval_label = np.random.randint(0, 10, batch_size)
        # eval_onehot = np.zeros((batch_size, 10))
        # eval_onehot[np.arange(batch_size), eval_label] = 1
        # eval_noise_[np.arange(batch_size), :10] = eval_onehot[np.arange(batch_size)]
        # eval_noise_ = (torch.from_numpy(eval_noise_))
        # eval_noise.data.copy_(eval_noise_.view(batch_size, 110, 1, 1))
        # eval_noise=eval_noise.cuda()

        optimD=torch.optim.Adam(self.discriminator.parameters(), lr)
        optimG=torch.optim.Adam(self.generator.parameters(), lr)

        source_obj=nn.BCELoss()     #source-loss
        class_obj=nn.NLLLoss()      #class-loss

        for epoch in range(epochs):
            for i,data in enumerate(dataloader,0):
                '''
                At first we will train the discriminator
                '''
                #training with real data----
                optimD.zero_grad()

                image,label = data
                image,label = image.cuda(),label.cuda()
                
                source_ = self.discriminator(image)                          #we feed the real images into the discriminator
                error_real = source_obj(source_,real_label)                  #label for real images--1; for fake images--0
                error_real.backward()
                optimD.step()

                #training with fake data now----

                noise_image = self.generate_fake(batch_size)
                source_ = self.discriminator(noise_image.detach())          #we will be using this tensor later on
                #print(source_.size())
                error_fake = source_obj(source_,fake_label)                 #label for real images--1; for fake images--0
                error_fake.backward()
                optimD.step()

                '''
                Now we train the generator as we have finished updating weights of the discriminator
                '''
                self.generator.zero_grad()
                source_ = self.discriminator(noise_image)
                error_gen = source_obj(source_,real_label)         #The generator tries to pass its images as real---so we pass the images as real to the cost function
                error_gen.backward()
                optimG.step()
                iteration_now = epoch * len(dataloader) + i

                if epoch%20==0:
                    logger.warning(f"***GAN training***: Epoch--[{epoch} / {epochs}], Loss_Discriminator--[{error_fake}], Loss_Generator--[{error_gen}]")

                # print("Epoch--[{} / {}], Loss_Discriminator--[{}], Loss_Generator--[{}],Accuracy--[{}]".format(epoch,epochs,error_fake,error_gen,accuracy))

                # '''Saving the images by the epochs'''
                # if i % 100 == 0:
                #     constructed = gen(eval_noise)
                #     vutils.save_image(
                #         constructed.data,
                #         '%s/results_epoch_%03d.png' % ('images/', epoch)
                #         )