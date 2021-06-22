import torch.utils.data as data_utils
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils

import torch
import torch.nn as nn

from models.gan import Generator, Discriminator

import copy
import logging
logger = logging.getLogger('logger')

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

class PDGAN():
    def __init__(self, hlpr=None, image_size=32, nc=3, nz=100, ngf=64, ndf=64, lr=.0002, beta1=.5):
	    self.hlpr = hlpr

	    self.netG = self.get_generator(nc, nz, ngf)
	    self.netG = self.resume_model(self.netG,"netG_")
	    self.netD = self.get_discriminator(nc, ndf)
	    self.netD = self.resume_model(self.netD,"netD_")
		
		# Setup Adam optimizers for both G and D
	    self.optimizerD = torch.optim.Adam(self.netD.parameters(), lr=lr, betas=(beta1, 0.999))
	    self.optimizerG = torch.optim.Adam(self.netG.parameters(), lr=lr, betas=(beta1, 0.999))
     
        # Initialize BCELoss function
	    self.criterion = nn.BCELoss()
     
	    self.image_size = image_size
	    self.latent_size = nz
	    #self.fixed_noise_list = []
	    self.aux_loader = None

    def get_generator(self, nc, nz, ngf):
		# Create the generator
	    self.netG = Generator(ngpu, nc, nz, ngf).to(device)
		# Handle multi-gpu if desired
	    if (device.type == 'cuda') and (ngpu > 1):
		    self.netG = nn.DataParallel(self.netG, list(range(ngpu)))

		# Apply the weights_init function to randomly initialize all weights
		#  to mean=0, stdev=0.2.
	    self.netG.apply(self.weights_init)

		# Print the model
	    #logger.warning(self.netG)

	    return self.netG

    def get_discriminator(self, nc, ndf):
		# Create the Discriminator
	    self.netD = Discriminator(ngpu, nc, ndf).to(device)
		# Handle multi-gpu if desired
	    if (device.type == 'cuda') and (ngpu > 1):
		    self.netD = nn.DataParallel(self.netD, list(range(ngpu)))

		# Apply the weights_init function to randomly initialize all weights
		#  to mean=0, stdev=0.2.
	    self.netD.apply(self.weights_init)

		# Print the model
	    #logger.warning(self.netD)

	    return self.netD
	
    def save_model(self, model:nn.Module=None, name=""):
        if self.hlpr.params.save_model:
            logger.info(f"Saving model to {self.hlpr.params.folder_path}.")
            model_name = '{}/model_{}weights.pth'.format(self.hlpr.params.folder_path, name)
            torch.save(model.state_dict(), model_name)

    def resume_model(self, model:nn.Module=None, name=""):
	    if self.hlpr.params.resume_model:
	        logger.info(f'Resuming training from {self.hlpr.params.resume_model}')
	        model_name = 'saved_models/{}/model_{}weights.pth'.format(self.hlpr.params.resume_model, name)
	        loaded_params = torch.load(model_name, map_location=torch.device('cpu'))
	        model.load_state_dict(loaded_params)
	        logger.warning(f"GAN model loaded.")
	    return model

    def weights_init(self, m):
	    classname = m.__class__.__name__
	    if classname.find("Conv") != -1:
		    nn.init.normal_(m.weight.data, 0.0, 0.02)
	    elif classname.find("BatchNorm") != -1:
		    nn.init.normal_(m.weight.data, 1.0, 0.02)
		    nn.init.constant_(m.bias.data, 0)

    def get_data_loader(self, batch_size, image_size):
		# We can use an image folder dataset the way we have it setup.
		# Create the dataset
	    dataset = dset.CIFAR10(root="dataset/cifar", download=True,
								transform=transforms.Compose([
									transforms.Resize(image_size),
									transforms.CenterCrop(image_size),
									transforms.ToTensor(),
									transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
								]))
		# Create the dataloader
	    self.aux_loader = data_utils.DataLoader(dataset, batch_size=batch_size,
											shuffle=True, num_workers=0)	

    def train(self, num_epochs=1, batch_size=64):
		# Establish convention for real and fake labels during training
	    real_label = 1.
	    fake_label = 0.

		# For each epoch
	    for epoch in range(num_epochs):
			# For each batch in the dataloader
		    for i, data in enumerate(self.aux_loader, 0):
				############################
				# (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
				###########################
				## Train with all-real batch
			    self.netD.zero_grad()
				# Format batch
			    real_cpu = data[0].to(device)
			    b_size = real_cpu.size(0)
			    label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
				# Forward pass real batch through D
			    output = self.netD(real_cpu).view(-1)
			    # Calculate loss on all-real batch
			    errD_real = self.criterion(output, label)
			    # Calculate gradients for D in backward pass
			    errD_real.backward()
			    D_x = output.mean().item()

				## Train with all-fake batch
				# Generate batch of latent vectors
			    noise = torch.randn(b_size, self.latent_size, 1, 1, device=device)
				# Generate fake image batch with G
			    fake = self.netG(noise)
			    label.fill_(fake_label)
			    # Classify all fake batch with D
			    output = self.netD(fake.detach()).view(-1)
			    # Calculate D's loss on the all-fake batch
			    errD_fake = self.criterion(output, label)
			    # Calculate the gradients for this batch, accumulated (summed) with previous gradients
			    errD_fake.backward()
			    D_G_z1 = output.mean().item()
			    # Compute error of D as sum over the fake and the real batches
			    errD = errD_real + errD_fake
			    # Update D
			    self.optimizerD.step()

				############################
				# (2) Update G network: maximize log(D(G(z)))
				###########################
			    self.netG.zero_grad()
			    label.fill_(real_label)  # fake labels are real for generator cost
			    # Since we just updated D, perform another forward pass of all-fake batch through D
			    output = self.netD(fake).view(-1)
			    # Calculate G's loss based on this output
			    errG = self.criterion(output, label)
			    # Calculate gradients for G
			    errG.backward()
			    D_G_z2 = output.mean().item()
			    # Update G
			    self.optimizerG.step()

				# Output training stats
			    if i % 50 == 0:
				    logger.warning('PDGAN Training. [%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
						% (epoch, num_epochs, i, len(self.aux_loader),
							errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

				# Save Losses for plotting later
			    #self.G_losses.append(errG.item())
			    #self.D_losses.append(errD.item())

			# Create batch of latent vectors that we will use to visualize
			#  the progression of the generator
		    fixed_noise = torch.randn(batch_size, self.latent_size, 1, 1, device=device)
		    with torch.no_grad():
			    fake = self.netG(fixed_noise).detach().cpu()
			    vutils.save_image(fake, '{}/fake_samples_epoch_{}.png'.format("./images", epoch), normalize=True)
    
    def run_defence(self, aux_loader, global_model, participant_updates, round_no):
        # PDGAN deactivated
        if self.hlpr.params.fl_pdgan == 0:
            return participant_updates

		# check if auxiliary data is loaded
        if self.aux_loader == None:
            self.aux_loader = aux_loader

		# Training Loop
        self.train()    # by default train only once
        self.save_model(model=self.netG, name="netG_")
        self.save_model(model=self.netD, name="netD_")

        #activate defence
        if round_no > self.hlpr.params.fl_pdgan:
            benign_update_list = []
            malicious_update_list = []

            # Perform accuracy auditings
            accuracy_list = []

            for k in range(len(participant_updates)):
                pdgan_hlpr_task = copy.deepcopy(self.hlpr.task)
                acc_list_k = []
                loss_list_k = []

                #generate fake images
                xfake = []
                for i in range(10):
                    # Generate batch of latent vectors
                    noise = torch.randn(self.hlpr.params.batch_size, self.latent_size, 1, 1, device=device)
                    # Generate fake image batch with G
                    with torch.no_grad():
                        fake = self.netG(noise).detach()
                        xfake.append(fake)
                
                #output a sample of fake
                vutils.save_image(fake, '{}/fake_samples_epoch_{:03d}_p{}.png'.format("./images", round_no, k), normalize=True)
                
                #initialise participant k's classification model
                updated_global_model_k = copy.deepcopy(global_model)
                weight_accumulator = pdgan_hlpr_task.get_empty_accumulator()
                pdgan_hlpr_task.accumulate_weights(weight_accumulator, participant_updates[k]['update'])
                pdgan_hlpr_task.update_global_model(weight_accumulator, updated_global_model_k)
            
                with torch.no_grad():
                    #Assign labels for x_fake based on L
                    for x_fake in xfake:
                        #get predicted outputs
                        pred = updated_global_model_k(x_fake)
                        _, labels = global_model(x_fake).topk(1, 1, True, True)
                        pdgan_hlpr_task.accumulate_metrics(outputs=pred, labels=torch.squeeze(labels))
                        accuracy = pdgan_hlpr_task.metrics[0].get_main_metric_value()
                        loss = pdgan_hlpr_task.metrics[1].get_main_metric_value()
                        acc_list_k.append(accuracy)
                        loss_list_k.append(loss)
                avg_acc = sum(acc_list_k)/len(acc_list_k)
                avg_loss = sum(loss_list_k)/len(loss_list_k)
                logger.warning(f"PDGAN. Participant {k}. Compromised {participant_updates[k]['user'].compromised}. Epoch: {round_no}. Accuracy: {avg_acc} | Loss: {avg_loss}")
                accuracy_list.append(avg_acc)

            #Calculate accuracy a of each participant classification model on x_fake
            mean_acc = sum(accuracy_list)/len(accuracy_list)
            logger.warning(f"mean_acc: {mean_acc}")
            acc_threshold = self.hlpr.params.fl_accuracy_threshold/100.0
            correct_purges = 0
            for k in range(len(accuracy_list)):
                if (accuracy_list[k] >= (1.0-acc_threshold)*mean_acc): #accuracy_low_threshold):# and metric <= accuracy_high_threshold):
                    benign_update_list.append(participant_updates[k])
                else:
                    malicious_update_list.append(participant_updates[k])
                    if participant_updates[k]['user'].compromised:
                        correct_purges +=1
            logger.warning(f"Number of correct participants purged: {correct_purges}/{len(malicious_update_list)}.")
            return benign_update_list
        else:
            return participant_updates