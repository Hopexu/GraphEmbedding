from torch.nn import Sequential as Seq,Linear,ReLU
from torch_scatter import scatter_mean
from .MetaLayer import *


class MyLayer(t.nn.Module):
    def __init__(self):
        super(MyLayer, self).__init__()

        self.edge_mlp = Seq(Linear(..., ...), ReLU(), Lin(..., ...))
        self.node_mlp_1 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
        self.node_mlp_2 = Seq(Lin(..., ...), ReLU(), Lin(..., ...))
        self.global_mlp = Seq(Lin(..., ...), ReLU(), Lin(..., ...))

        def edge_model(src, dest, edge_attr, u, batch):
            # source, target: [E, F_x], where E is the number of edges.
            # edge_attr: [E, F_e]
            # u: [B, F_u], where B is the number of graphs.
            # batch: [E] with max entry B - 1.
            out = t.cat([src, dest, edge_attr, u[batch]], 1)
            return self.edge_mlp(out)

        def node_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            row, col = edge_index
            out = t.cat([x[col], edge_attr], dim=1)
            out = self.node_mlp(out)
            out = scatter_mean(out, row, dim=0, dim_size=x.size(0))
            out = t.cat([out, u[batch]], dim=1)
            return self.node_mlp_2(out)
'''
        def global_model(x, edge_index, edge_attr, u, batch):
            # x: [N, F_x], where N is the number of nodes.
            # edge_index: [2, E] with max entry N - 1.
            # edge_attr: [E, F_e]
            # u: [B, F_u]
            # batch: [N] with max entry B - 1.
            out = t.cat([u, scatter_mean(x, batch, dim=0)], dim=1)
            return self.global_mlp(out)
'''
        self.op = MetaLayer(edge_model, node_model, None)

    def forward(self, x, edge_index, edge_attr, u, batch):
        return self.op(x, edge_index, edge_attr, u, batch)
