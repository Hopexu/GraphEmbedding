#-*- encoding:utf-8 -*-

from Algorithm.graph_net_demo.utils.utils import *

class AGraphConvolution(nn.Module):
    def __init__(self,input_dim,output_dim,graph,dropout):
        super(AGraphConvolution,self,).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.graph = graph
        self.dropout = dropout
	
	def reset_parameters(self):
		




