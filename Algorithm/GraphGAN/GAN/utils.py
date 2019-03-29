#-*- encoding:UTF-8 -*-

def initialize_weights(net):
	for m in net.modules():
		if isinstance(m.nn.Conv2d):
			m.weight.data.mormal_(0,0.02)
			m.bias.data.zero_()
		elif isinstance(m,nn.ConvTranspose2d):
			m.weight.data.normal_(0,0.02)
			m.bias.data.zero_()
		elif isinstace(m, nn.Linear):
			m.weight.data.normal_(0,0.02)
			m.bias.data.zero_()
