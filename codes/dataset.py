import torch
import networkx as nx
import numpy as np
import pickle as pkl
import scipy.sparse as sp
import torch.utils.data
import itertools
from collections import Counter
from random import shuffle
import json
#
from networkx.readwrite import json_graph
from argparse import ArgumentParser
import matplotlib.pyplot as plt

import pdb
import time
import random
import pickle
import os.path
import torch_geometric as tg
import torch_geometric.datasets
import time

from torch_geometric.data import Data, DataLoader

def nx_to_tg_data(graphs, features, pair_nodes):
    data_list = []
    for i in range(len(graphs)):
        feature = features[i]
        graph = graphs[i].copy()
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # relabel graphs
        keys = list(graph.nodes)
        vals = range(graph.number_of_nodes())
        mapping = dict(zip(keys, vals))
        nx.relabel_nodes(graph, mapping, copy=False)

        x = np.zeros(feature.shape)
        graph_nodes = list(graph.nodes)
        for m in range(feature.shape[0]):
            x[graph_nodes[m]] = feature[m]
        x = torch.from_numpy(x).float()

        # get edges
        edge_index = np.array(list(graph.edges))
        edge_index = np.concatenate((edge_index, edge_index[:,::-1]), axis=0)
        edge_index = torch.from_numpy(edge_index).long().permute(1,0)

        data = Data(x=x, edge_index=edge_index)

        data.pair_nodes_class = pair_nodes[i]
        data_list.append(data)

    return data_list


def Graph_load_batch(name, min_num_nodes = 20, max_num_nodes = 10000, node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'PROTEINS_full/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # split into graphs
    graph_num = data_graph_indicator.max()
    print("Graph Nums {}".format(graph_num))
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
        
    print('Loaded {} graphs having node sizes in permissible limits'.format(len(graphs)))
    return graphs, data_node_att, data_node_label


# main data load function
def load_protein_graph():
    
    same_labels = []
    
    
    graphs_all, features_all, labels_all = Graph_load_batch(name='PROTEINS_full')
        
    features_all = (features_all-np.mean(features_all,axis=-1,keepdims=True))/np.std(features_all,axis=-1,keepdims=True)
    
    graphs = []
    features = []

    for graph in graphs_all:
        n = graph.number_of_nodes()
        label = np.zeros((n, n),dtype=int)
        for i,u in enumerate(graph.nodes()):
            for j,v in enumerate(graph.nodes()):
                if labels_all[u-1] == labels_all[v-1]:
                    label[i,j] = 1

        graphs.append(graph)

        same_labels.append(label)

        idx = [node-1 for node in graph.nodes()]
        feature = features_all[idx,:]
        features.append(feature)


    return graphs, features, same_labels


def load_protein_dataset():
    graphs, features, pair_nodes = load_protein_graph()
    return nx_to_tg_data(graphs, features, pair_nodes)