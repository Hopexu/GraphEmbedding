#-*- encoding:utf-8 -*-

from .blocks.blocks import *

DEFAULT_EDGE_BLOCK_OPT = {
    'use_edges':True,
    'use_receive_nodes':True,
    'use_sender_nodes':True,
    'use_globals':True
}

DEFAULT_NODE_BLOCK_OPT = {
    'use_receive_edges':True,
    'use_send_edges':False,
    'use_nodes':True,
    'use_globals':True
}

DEFAULT_GLOBAL_BLOCK_OPT = {
    "use_edges": True,
    "use_nodes": True,
    "use_globals": True,
}

def make_default_edge_block_opt(edge_block_opt):
    edge_block_opt = dict(edge_block_opt.items()) if edge_block_opt else {}
    for k,v in DEFAULT_EDGE_BLOCK_OPT.items():
        edge_block_opt[k] = edge_block_opt.get(k,v)
    return edge_block_opt

def make_default_node_block_opt(node_block_opt,default_reducer):
    node_block_opt = dict(node_block_opt.items()) if node_block_opt else {}
    for k,v in DEFAULT_NODE_BLOCK_OPT.items():
        node_block_opt[k] = node_block_opt.get(k,v)
    for key in ["received_edges_reducer", "sent_edges_reducer"]:
        node_block_opt[key] = node_block_opt.get(key, default_reducer)
    return node_block_opt

def make_default_gobal_opt(global_block_opt,default_reducer):
    global_block_opt = dict(global_block_opt.items()) if global_block_opt else {}
    for k,v in DEFAULT_GLOBAL_BLOCK_OPT.items():
        global_block_opt[k] = global_block_opt.get(k,v)
    for key in ["edges_reducer", "nodes_reducer"]:
        global_block_opt[key] = global_block_opt.get(key, default_reducer)
    return global_block_opt

class GraphNetwork():
    def __init__(self,
                 edge_model_fn,
                 node_model_fn,
                 global_model_fn,
                 reducer=None,
                 edge_block_opt=None,
                 node_block_opt=None,
                 global_block_opt=None,
                 name="graph_network"):
        super(GraphNetwork,self).__init__()
        edge_block_opt = make_default_edge_block_opt(edge_block_opt)
        node_block_opt = make_default_node_block_opt(node_block_opt,reducer)
        global_node_opt = make_default_gobal_opt(global_block_opt,reducer)


