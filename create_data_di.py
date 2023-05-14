import os
import random
import networkx as nx
import dgl
import math
import numpy as np
from tqdm import tqdm
from generate_data.topology2conflict import generate_havel_hakimi_graph, shuffle, t2c_directed, t2c_undirected, generate_random_bigraph, add_interference
from generate_data.Run_exact_di_timelimit import run_exact
from generate_data import Matlab

def generate_directed_er_graph(n, p, directed=True):
    G = dgl.DGLGraph()
    G.add_nodes(n)

    w = -1
    lp = math.log(1.0 - p)
    edges_list = []
    v = 0
    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        if v == w:
            w = w + 1
        while v < n <= w:
            w = w - n
            v = v + 1
            if v == w:
                w = w + 1
        if v < n:
            if directed:
                edges_list.extend([(v, w)])
            else:
                edges_list.extend([(v, w), (w, v)])

    edges_list = list(set(edges_list))
    if edges_list != []:
        G.add_edges(*zip(*edges_list))
    return G

from networkx.utils import open_file
@open_file(1, mode="wb")
def write_edgelist(G, path, delimiter=" ", data=True, encoding="utf-8"):
    info = "p edge {} {}".format(G.number_of_nodes(), G.number_of_edges())
    info += "\n"
    path.write(info.encode(encoding))
    for line in nx.readwrite.edgelist.generate_edgelist(G, delimiter, data):
        line = "e " + line
        line += "\n"
        path.write(line.encode(encoding))
    return

if __name__ == "__main__":
    # set the graph type you want to generate and the corresponding parameters
    graph_type = 'random_bigraph'
    num_graph = 100
    #random ER bipartite
    num_nodes_left_er = 20
    num_nodes_right_er = 20
    p_bi = 0.2
    q_bi = 0.2
    #random HH bipartite
    num_nodes_HH = 30
    max_degree_HH = 8
    q_HH = 0.2
    #random geo graph
    num_nodes_geo = 50
    radius = 0.2
    #random ba graph
    num_nodes_ba=50
    m_ba=7
    #random network
    num_t, num_antenna, p_network = 8, 1, 0.3
    num_r = num_t
    #random PA_graph
    num_nodes_pre = 30
    max_degree_pre = 6
    p_pre = 0.2
    q_pre = 0.2
    ######################################################\

    print(graph_type)
    if graph_type == 'random_bigraph':
        base_data_dir = os.path.join('data/data_di_bipar_{}{}{}{}_{}_test'
                                 .format(num_nodes_left_er, num_nodes_right_er, p_bi, q_bi, num_graph))
    elif graph_type == 'random_bipartite_HH':
        base_data_dir = os.path.join('../data/data_di_biparHH_{}{}{}{}_{}'
                                     .format(num_nodes_HH, num_nodes_HH, max_degree_HH, q_HH, num_graph))
    elif graph_type == 'geo_graph':
        base_data_dir = os.path.join('../data/data_di_geo_{}{}_{}'
                                     .format(num_nodes_geo, radius, num_graph))
    elif graph_type == 'ba_graph':
        base_data_dir = os.path.join('../data/data_di_ba_{}{}_{}'
                                 .format(num_nodes_ba, m_ba, num_graph))
    elif graph_type == 'network':
        base_data_dir = os.path.join('../data/data_di_network_{}{}{}{}_txt_{}'
                                     .format(num_t, num_r, num_antenna, p_network, num_graph))
    elif graph_type == 'PA_graph':
        base_data_dir = os.path.join('../data/data_di_preferential_{}{}{}{}'
                                     .format(num_nodes_pre,max_degree_pre,p_pre, q_pre, num_graph))
    else:
        print('invalid name')



    C_list = []
    data_dir_undi = os.path.join(base_data_dir,"undirected")
    data_dir_di = os.path.join(base_data_dir,"directed")
    os.makedirs(data_dir_undi, exist_ok = True)
    os.makedirs(data_dir_di, exist_ok = True)

    for g_idx in tqdm(range(num_graph)):
        if graph_type == 'geo_graph':
            G_undi = nx.random_geometric_graph(n = num_nodes_geo, radius = radius)
        elif graph_type == 'ba_graph':
            G_undi = nx.generators.random_graphs.barabasi_albert_graph(n = num_nodes_ba, m = m_ba)
        elif graph_type == 'random_bigraph':
            G = generate_random_bigraph(num_nodes_left_er, num_nodes_right_er, p_bi, q_bi)
            g_dgl_undi = t2c_undirected(G)
            g_dgl_di = t2c_directed(G)
            G_undi = g_dgl_undi.to_networkx()
            G_di = g_dgl_di.to_networkx()
        elif graph_type == 'random_bipartite_HH':
            aseq1 = [random.randint(1, max_degree_HH) for _ in range(num_nodes_HH)]
            bseq1 = shuffle(aseq1)
            G = generate_havel_hakimi_graph(aseq1, bseq1,q_HH)
            g_dgl_undi = t2c_undirected(G)
            G_undi = g_dgl_undi.to_networkx()
        elif graph_type == 'network':
            G = Matlab.generate_network(num_t, num_r, num_antenna, p_network)
            g_dgl_undi = t2c_undirected(G)
            G_undi = g_dgl_undi.to_networkx()
            g_dgl_di = t2c_directed(G)
            G_di = g_dgl_di.to_networkx()
        elif graph_type == 'PA_graph':
            aseq = [random.randint(1, max_degree_pre) for _ in range(num_nodes_pre)]
            G = nx.bipartite.preferential_attachment_graph(aseq, p_pre, create_using=nx.Graph, seed=None)
            G = add_interference(G, q_pre)
            g_dgl_undi = t2c_undirected(G)
            G_undi = g_dgl_undi.to_networkx()
        else:
            print("invalid name")




        G_undi = nx.convert_node_labels_to_integers(G_undi, first_label=1)
        G_di = nx.convert_node_labels_to_integers(G_di, first_label=1)

        nx_g_path_undi = os.path.join(data_dir_undi, "{:06d}.txt".format(g_idx))
        nx_g_path_di = os.path.join(data_dir_di, "{:06d}.txt".format(g_idx))
        write_edgelist(G_undi, path = nx_g_path_undi, data=False)
        write_edgelist(G_di, path=nx_g_path_di, data=False)

    run_exact(base_data_dir)
