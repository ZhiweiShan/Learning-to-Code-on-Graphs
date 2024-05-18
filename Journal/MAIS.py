import matplotlib
import matplotlib.pyplot as plt
import os
import random
import networkx as nx
import math
import numpy as np
from networkx.readwrite.edgelist import read_edgelist
import pickle
import helper_functions

#matplotlib.use('Qt5Agg')
from itertools import combinations
import sys

def is_acyclic(graph):
    try:
        nx.find_cycle(graph)
        return False
    except nx.NetworkXNoCycle:
        return True

def complement_MAIS_size(graph):
    graph = nx.complement(graph)
    nodes = list(graph.nodes)
    for r in range(len(nodes), 0, -1):
        for subset in combinations(nodes, r):
            subgraph = graph.subgraph(subset)
            # nx.draw(subgraph, pos=nx.circular_layout(subgraph), with_labels=True, arrows=True, node_size=500,
            #         node_color='lightblue')
            # plt.title("Maximal Acyclic Induced Subgraph")
            # plt.show()
            if nx.is_directed_acyclic_graph(subgraph):
                # nx.draw(subgraph, pos=nx.circular_layout(subgraph), with_labels=True, arrows=True, node_size=500,
                #         node_color='lightblue')
                # plt.title("Maximal Acyclic Induced Subgraph")
                # plt.show()
                return len(subgraph)
    return 0  # 如果没有找到无环诱导子图，则返回0

# Example usage:
# n = 6 # 节点数量
# p = 0.4  # 边的概率
#
# G = nx.gnp_random_graph(n, p, directed=True)
# nx.draw(G, pos=nx.circular_layout(G), with_labels=True, arrows=True, node_size=500, node_color='lightblue')
# plt.title("Origin Graph")
# plt.show()
#
# max_size = complement_maximal_acyclic_subgraph_size(G)
# print("Origin Graph size:", G.number_of_nodes())
# print("Size of Maximal acyclic induced subgraph:", max_size)
if __name__ == '__main__':

    data_dir_base = '../data/data_di_bipar_550.40.4_5000_test'
    try:
        data_dir_base = sys.argv[1:][0]
    except:
        pass
    data_dir_di = os.path.join(data_dir_base, 'directed')
    num_eval_graphs = len([
        name
        for name in os.listdir(data_dir_di)
        if name.endswith('.txt')
    ])
    result_dir = os.path.join(data_dir_base,'MAIS.pickle')
    upper_dof_index = {}
    for idx in range(num_eval_graphs):
        g_path = os.path.join(data_dir_di,
            "{:06d}.txt".format(idx)
        )
        g = read_edgelist(g_path, create_using=nx.DiGraph)
        MAIS = complement_MAIS_size(g)
        key = '1-{}'.format(MAIS)
        if key not in upper_dof_index:
            upper_dof_index[key] = []
        upper_dof_index[key].append(idx)
    print('MAIS')
    helper_functions.sort_dict(upper_dof_index)

    with open(result_dir, 'wb') as f:
        pickle.dump(upper_dof_index, f)
