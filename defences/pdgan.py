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
import torchvision.utils as vutils
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import copy

from models.gan import Generator, Discriminator

import logging
logger = logging.getLogger('logger')

class PDGAN():
    def __init__(self):
        os.makedirs("images", exist_ok=True)

        self.generator = None
        self.discriminator = None
    
    def compute_acc(self, preds, labels):
        correct = 0
        correct = preds.eq(labels.data).cpu().sum()
        acc = float(correct) / float(len(labels.data)) * 100.0
        return acc

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def _to_var(self, x):
        '''
        Creates a variable for a tensor
        :param x: PyTorch Tensor
        :return: Variable that wraps the tensor
        '''
        x = x.cuda()
        return Variable(x)

    def run_defence(self, hlpr, aux_loader, global_model, participant_updates, round_no, lr=0.0002, batch_size=100):
        if hlpr.params.fl_pdgan == 0:
            return participant_updates
            
        benign_update_list = []
        malicious_update_list = []
        # # Initialize generator and discriminator if not exist
        if self.generator == None:
            g = Generator(100, 64, .2, 3,True)
            g.apply(self.weights_init)
            self.generator = g.cuda()
        
        # update discriminator
        updated_global_model = copy.deepcopy(global_model)
        self.discriminator = Discriminator(updated_global_model, num_classes=10).cuda()

        #train generator and discriminator
        self.train(aux_loader, batch_size, round_no, lr)
        
        if round_no > hlpr.params.fl_pdgan:
            ##Perform accuracy auditing
            accuracy_list = []
            for k in range(len(participant_updates)):
                pdgan_hlpr_task = copy.deepcopy(hlpr.task)
                acc_list_k = []
                loss_list_k = []

                #generate then load fake
                xfake = []
                for i in range(100):
                    fake = self.generate_fake()
                    vutils.save_image(fake.data, '{}/fake_samples_epoch_{:03d}.png'.format("./images", round_no), normalize=True)
                    xfake.append(fake)
                #xfake_loader = DataLoader((x_fake,global_model(x_fake)), batch_size=hlpr.params.batch_size, shuffle=True, num_workers=0)
            
                #initialise participant classification model
                updated_global_model_k = copy.deepcopy(global_model)
                weight_accumulator = pdgan_hlpr_task.get_empty_accumulator()
                pdgan_hlpr_task.accumulate_weights(weight_accumulator, participant_updates[k]['update'])
                pdgan_hlpr_task.update_global_model(weight_accumulator, updated_global_model_k)
                
                with torch.no_grad():
                    #Assign labels for x_fake based on L
                    for x_fake in xfake:
                        outputs = updated_global_model_k(x_fake)
                        _, labels = global_model(x_fake).topk(1, 1, True, True)
                        pdgan_hlpr_task.accumulate_metrics(outputs=outputs, labels=torch.squeeze(labels))
                        accuracy = pdgan_hlpr_task.metrics[0].get_main_metric_value()
                        loss = pdgan_hlpr_task.metrics[1].get_main_metric_value()
                        #logger.warning(f"Accuracy: {accuracy} | Loss: {loss}")
                        acc_list_k.append(accuracy)
                        loss_list_k.append(loss)
                mean_acc = sum(acc_list_k)/len(acc_list_k)
                mean_loss = sum(loss_list_k)/len(loss_list_k)
                logger.warning(f"PDGAN. Participant {k}. Compromised {participant_updates[k]['user'].compromised}. Epoch: {round_no}. Accuracy: {mean_acc} | Loss: {mean_loss}")
                accuracy_list.append(mean_acc)
            
            #Calculate accuracy a of each participant classification model on x_fake
            mean_accuracy = sum(accuracy_list)/len(accuracy_list)
            accuracy_threshold = hlpr.params.fl_accuracy_threshold/100.0
            correct_purges = 0
            for k in range(len(accuracy_list)):
                if (accuracy_list[k] >= (1.0-accuracy_threshold)*mean_accuracy): #accuracy_low_threshold):# and metric <= accuracy_high_threshold):
                    benign_update_list.append(participant_updates[k])
                else:
                    malicious_update_list.append(participant_updates[k])
                    if participant_updates[k]['user'].compromised:
                        correct_purges +=1
            logger.warning(f"Number of correct participants purged: {correct_purges}/{len(malicious_update_list)}.")
            return benign_update_list
        else:
            return participant_updates
    
    def assign_labels(self, outputs):
        #n output tensors for n participants
        return torch.mode(outputs,0).values

    def generate_fake(self):
        #noise_ = np.random.normal(0, 1, (batch_size, 100))        #generating noise by random sampling from a normal distribution
        #label = np.random.randint(0,10,batch_size)                #generating labels for the entire batch
        
        #noise = ((torch.from_numpy(noise_)).float())
        #noise = noise.cuda()                                      #converting to tensors in order to work with pytorch

        #label = ((torch.from_numpy(label)).long())
        #label = label.cuda()                                      #converting to tensors in order to work with pytorch
        #return self.generator(noise)
        noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
        noise_var = self._to_var(noise)
        return self.generator(noise_var)

    def train(self, dataloader, batch_size, round_no, lr):
        d_optimizer=torch.optim.Adam(self.discriminator.parameters(), .0002, betas=(.5, 0.999))
        g_optimizer=torch.optim.Adam(self.generator.parameters(), .0002, betas=(.5, 0.999))

        self.discriminator.train()
        self.generator.train()
        
        gloss_sum = .0
        gloss_num = .0
        dloss_sum = .0
        dloss_num = .0
        for batch_id,batch in enumerate(dataloader,0):
            #training with real data----
            image,label = batch
            image = image.cuda()
            label = label.cuda()
            d_optimizer.zero_grad()           
            
            d_out, d_class_logits_on_data, d_gan_logits_real = self.discriminator(image)
            d_gan_labels_real = torch.LongTensor(64)
            d_gan_labels_real.resize_as_(label.data.cpu()).fill_(1)
            d_gan_labels_real_var = self._to_var(d_gan_labels_real).float()
            d_gan_criterion = nn.BCEWithLogitsLoss()
            d_gan_loss_real = d_gan_criterion(d_gan_logits_real, d_gan_labels_real_var)

            #training with fake data now----
            fake = self.generate_fake()
            # call detach() to avoid backprop for G here   
            _, _, d_gan_logits_fake = self.discriminator(fake.detach())          #we will be using this tensor later on
            
            d_gan_labels_fake = torch.LongTensor(64).resize_(64).fill_(0)
            d_gan_labels_fake_var = self._to_var(d_gan_labels_fake).float()
            d_gan_criterion = nn.BCEWithLogitsLoss()
            d_gan_loss_fake = d_gan_criterion(d_gan_logits_fake, d_gan_labels_fake_var)

            d_loss = d_gan_loss_real + d_gan_loss_fake
            dloss_sum += d_loss
            dloss_num += 1
            
            d_loss.backward()
            d_optimizer.step()

            d_optimizer.zero_grad()
            g_optimizer.zero_grad()

            # train with fake images
            # noise = torch.FloatTensor(64, 100, 1, 1).normal_(0, 1)
            # noise_var = _to_var(noise)
            # fake = helper.g_model(noise_var)

            '''
            Now we train the generator as we have finished updating weights of the discriminator
            '''
            _, _, g_gan_logits = self.discriminator(fake)

            g_loss = -torch.mean(g_gan_logits)
            gloss_sum += g_loss
            gloss_num += 1
            g_loss.backward()
            g_optimizer.step()

            logger.warning(f"***GAN training***: Epoch: [{round_no}]. Loss_Discriminator--[{dloss_sum / dloss_num}]. Loss_Generator--[{gloss_sum / gloss_num}].")