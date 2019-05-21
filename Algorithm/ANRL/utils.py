import networkx as nx
import numpy as np
import collections
import json
import time
import numpy as np
import pandas as pd
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Linear,Sequential,ReLU,ModuleList,Parameter

def read_param(fname):
    with open(fname) as f:
        data = json.load(f)
        return [data]


def read_graph(edgeFile):
    print('loading graph...')
    G = nx.read_edgelist(edgeFile, nodetype=int, create_using=nx.DiGraph())
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    return G


def read_edgelist(inputFileName):
    f = open(inputFileName, 'r')
    lines = f.readlines()
    f.close()

    edgelist = []
    for line in lines:
        l = line.strip('\n\r').split(' ')
        edge = (int(l[0]), int(l[1]))
        edgelist.append(edge)
    return edgelist


def read_feature(inputFileName):
    f = open(inputFileName, 'r')
    lines = f.readlines()
    f.close()

    features = []
    for line in lines[1:]:
        l = line.strip('\n\r').split(' ')
        features.append(l)
    features = np.array(features, dtype=np.float32)

    return features


def write_embedding(embedding_result, outputFileName):
    f = open(outputFileName, 'w')
    N, dims = embedding_result.shape

    for i in range(N):
        s = ''
        for j in range(dims):
            if j == 0:
                s = str(i) + ',' + str(embedding_result[i, j])
            else:
                s = s + ',' + str(embedding_result[i, j])
        f.writelines(s + '\n')
    f.close()
