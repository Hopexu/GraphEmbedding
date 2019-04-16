#-*- encoding:utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from graph_nets import blocks
from graph_nets import graphs
from graph_nets import modules
from graph_nets import utils_np
from graph_nets import utils_tf

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sonnet as snt
import tensorflow as tf

def name2id(filepath):
    namedict = {}
    with open(filepath) as f:
        for line in f:
            s = line.strip().split('\t')
            namedict[s[0]] = int(s[1])
    return namedict

def readtriple(trainfile,testfile):
    entity_set = set()
    realtion_set = set()

    return h_t_rdict
if __name__ == '__main__':
    #attr2id = name2id('../../../data/h_r_t_attr/attribute2id.txt')
    #entity2id = name2id('../../../data/h_r_t_attr/entity2id.txt')
    #relation2id = name2id('../../../data/h_r_t_attr/relation2id.txt')
    h_t_rdict = readtriple('../../../data/h_r_t_attr/train-attr.txt',)
    print(h_t_rdict)
    print(attr2id)
    print(entity2id)
    print(relation2id)