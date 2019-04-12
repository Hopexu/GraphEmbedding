#-*- encoding:UTF-8 -*-
from Algorithm.GraphGAN.GAN.utils import *
	
class generator(nn.Module):
	def __init__(self,dataset = 'mnist'):
		super(generator,self).__init__()
		if dataset == 'mnist' or dataset == 'fashion-mnist':
			self.input_height = 28
			self.input_width = 28
			self.input_dim = 62
			self.output_dim = 1
		elif dataset == 'celebA':
			self.input_height = 64
			self.input_width = 64
			self.input_dim = 62
			self.output_dim = 3
		self.fc = nn.Sequential(
				nn.Linear(self.input_dim,1024),
				nn.BatchNorm1d(1024),
				nn.ReLU(),
				nn.Linear(1024,128*(self.input_height //4)*(self.input_width //4)),
				nn.BatchNorm1d(128 * (self.input_height //4) * (self.input_width //4)),
				nn.ReLU(),
				)
		self.decov = nn.Sequential(
				nn.ConvTranspose2d(128,64,4,2,1),
				nn.BatchNorm2d(64),
				nn.ReLU(),
				nn.ConvTranspose2d(64,self.output_dim,4,2,1),
				nn.Sigmoid(),
				)
		initialize_weights(self)
	def forward(self,input):
		pass
				
class discriminator(nn.Module):
	def __init__(self,dataset = 'mnist'):
		super(discriminator,self).__init__()
		if dataset == 'mnist' or dataset == 'fashion-mnist':
			self.input_height = 28
			self.input_width = 28
			self.input_dim = 1
			self.output_dim = 1
		elif dataset == 'celebA':
			self.input_height = 64
			self.input_width = 64
			self.input_dim = 3
			self.output_dim = 1
		self.fc = nn.Sequential(
				nn.Linear(128*(self.input_height //4)*(self.input_width //4),1024),
				nn.BatchNorm1d(1024),
				nn.LeakyReLU(0.2),
				nn.Linear(1024,self.output_dim),
				nn.Sigmoid(),
				)
		self.conv = nn.Sequential(
				nn.ConvTranspose2d(self.input_dim,64,4,2,1),
				nn.LeakyReLU(0.2),
				nn.Conv2d(64,128,4,2,1),
				nn.BatchNorm2d(128),
				nn.LeakyReLU(0.2),
				)
		utils.initializ_weithts(self)
	def forward(self,input):
		x = self.conv(input)
		x = x.view(-1,128*(self.input_height //4)*(self.input_width //4))
		x = self.fc(x)
		return x
