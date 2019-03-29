#-*- encoding:-8 -*-


def parse_args():
	desc = 'pytorch implementation of GAN collections'
	parser = argparse.ArgumentParser(description = desc)
	parser.add_argument('--gan_type',type = str, default = 'GAN',
				choices =['GAN','CGAN','infoGAN','ACGAN','EBGAN','BEGAN','WGAN','WGAN_GP'],
				help = 'The type of GAN')
	parser.add_argument('--dataset',type = str,default = 'mnist',
				choices = ['mnist','fashion-mnist'],
				help = 'The name of dataset')
	parser.add_argument('--epoch',type = int,default = 25,help 'The number of epochs to run')
	parser.add_argument('--batch_size',type = int,default = 64,help = 'The size of batch')
	parser.add_argument('--save_dir',type = str,default = 'models',help = 'Directory name to save the model')
	parser.add_argument('-result_dir',type = str,default = 'results',help = 'Directory name to save the generated images')
	parser.add_argument('--log_dir',type = str,default = 'logs',help = 'Directory name to save the logs')
	parser.add_argumetn('--lrG',type = float,default = 0.002)
	parser.add_argument('--lrD',type = float,default = 0.002)
	parser.add_argument('--beta1',type = float,default = 0.5)	
	parser.add_argument('--beta2',type = float,default = 0.99)
	parser.add_argument('--gpu_model',type = bool,default = True)
	return check_args(parser)
def main():
	args = parse_arg()
	if args is None:
		exit()
	if args.gan_type == 'GAN':
		gan = GAN(args)
if __name__ == '__main__':
	main()
