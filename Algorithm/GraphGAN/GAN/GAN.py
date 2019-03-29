#-*- encoding:UTF-8 -*-

class GAN(object):
	def __init__(self,args):
		self.epoch = args.epoch
		self.sample_num = 16
		self.batch_size = args.batch_size
		self.save_dir = args.save_dir
		self.results = args.results
		self.dataset = args.dataset
		self.log_dir = args.log_dir
		self.gpu_mode = args.gpu_mode
		self.model_name = args.gan_type
		self.G = generator(self.dataset)	
		self.D = discriminator(self.dataset)
		self.G_optimizer = optim.Adam(self.G.parameters(),lr = args.lrG,betas = (args.beta1,args.
