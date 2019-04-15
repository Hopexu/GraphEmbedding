#-*- encoding;utf-8 -*-

import torch
from torch.nn import Sequential as Seq,Linear,ReLU
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer


class Net(torch.nn.Mdoule):
    def __init__(self):
        super(Net,self).__init__()

        self.edge_mlp = Seq(Linear(128,256),ReLU(),Linear(256,128))
        self.node_mlp = Seq(Linear(128,256),ReLU(),Linear(256,128))
        self.global_mlp = Seq(Linear(128,256).ReLU(),Linear(256,128))

        def edge_model(source,target,edge_attr,u):
            out = torch.cat([source,target,edge_attr],dim = 1)
            return self.edge_mlp(out)

        def node_model(x,edge_index,edge_attr,u):
            row,col = edge_index
            out = torch.cat([x[col],edge_attr],dim = 1)
            out = self.node_mlp(out)
            return scatter_mean(out,row,dim = 0,dim_size = x.size(0))

        def global_model(x,edge_index,edge_attr,u,batch):
            out = torch.cat([u,scatter_mean(x,batch,dim = 0)],dim = 1)
            return self.global_mlp(out)

        self.op = MetaLayer(edge_model,node_model,global_model)

    def forward(self,x,edge_index,edge_attr,u,batch):
        return self.op(x,edge_index,edge_attr,u,batch)