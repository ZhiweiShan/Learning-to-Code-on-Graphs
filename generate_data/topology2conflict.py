import random
import networkx as nx
import dgl
import math
from copy import deepcopy
from random import randint
import numpy as np

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
        if v == w:  #
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
    G.add_edges(*zip(*edges_list))
    return G

def generate_havel_hakimi_graph(aseq1, bseq1,q):
    # q is the probability that an edge is demanded
    G = nx.algorithms.bipartite.generators.havel_hakimi_graph(aseq1, bseq1, create_using=nx.Graph())
    G = add_interference(G,q)
    return G
def generate_random_triangle_graph (n=10):
    #p is the prb. of an edge is both directed
    deg = []
    for u in range(n):
        d_ui = randint(0,1)
        d_ut = randint(1,1)
        deg.append((d_ui,d_ut))
    if np.sum(deg,0)[0]% 2 != 0:
        deg[0] = (deg[0][0]+1,deg[0][1])
    while np.sum(deg,0)[1]% 3 != 0:
        deg[0] = (deg[0][0], deg[0][1]+1)
    G = nx.random_clustered_graph(deg)
    G = nx.Graph(G)
    G.remove_edges_from(nx.selfloop_edges(G))

    return G

def remove_some_direction(G,p=0.2):
    #random remove some direction:
    #first, randomly add one direction; then add another direction with prb. p
    G_dgl = dgl.DGLGraph()
    G_dgl.add_nodes(len(list(G.nodes)))
    for (u, v) in list(G.edges):
        x = random.uniform(0, 1)
        if x > 0.5:
            G_dgl.add_edges(u, v)
            y = random.uniform(0, 1)
            if y <= p:
                G_dgl.add_edges(v, u)
        else:
            G_dgl.add_edges(v, u)
            y = random.uniform(0, 1)
            if y <= p:
                G_dgl.add_edges(u, v)
    return G_dgl

def generate_random_bigraph (m=3,n=4,p=0.5,q=0.5):
    #p is the probability that an edge exists, q is the probability that an edge is demanded
    G = nx.algorithms.bipartite.random_graph(m, n, p)
    nx.set_edge_attributes(G, 1, 'demand')
    index = random.sample(range(0, len(list(G.edges))), int(len(list(G.edges)) * (1-q)))
    diction = {list(G.edges)[index[i]]: 2 for i in range(len(index))}
    nx.set_edge_attributes(G, diction, 'demand')
    return G

def add_interference(G,q=0.5):
    nx.set_edge_attributes(G, 1, 'demand')
    index = random.sample(range(0, len(list(G.edges))), int(len(list(G.edges)) * (1 - q)))
    diction = {list(G.edges)[index[i]]: 2 for i in range(len(index))}
    nx.set_edge_attributes(G, diction, 'demand')
    return G

def draw_bigraph(G):
    top = nx.bipartite.sets(G)[0]
    pos = nx.bipartite_layout(G, top)
    nx.draw(G, pos=pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos=pos)
    return

def remove_interference_node(G_line_square,G):
    edge_attributes = nx.get_edge_attributes(G, 'demand')
    interference_list = list({n for n, a in edge_attributes.items() if a == 2})
    G_line_square.remove_nodes_from(interference_list)
    G_line_square.remove_nodes_from(list(map(tuple, np.array(interference_list)[:, [1, 0]])))
    return G_line_square

def decided_direction(G,G_line_demand):
    G_final = nx.DiGraph(G_line_demand)
    Edges = list(G_final.edges)
    for (a,b),(c,d) in Edges:
        if a==c or b==d:
            continue
        if not G.has_edge(a,d):
            if G_final.has_edge((a, b), (c, d)):
                G_final.remove_edge((a, b), (c, d))
        if not G.has_edge(c,b):
            if G_final.has_edge((c, d), (a, b)):
                G_final.remove_edge((c, d), (a, b))
    return G_final

def t2c_directed(G):
    G_line = nx.generators.line.line_graph(G)
    G_line_square = nx.algorithms.operators.product.power(G_line, 2)
    G_line_demand = remove_interference_node(G_line_square,G)
    G_final = decided_direction(G, G_line_demand)
    g = dgl.from_networkx(G_final)
    return g

def t2c_undirected(G):
    G_line = nx.generators.line.line_graph(G)
    G_line_square = nx.algorithms.operators.product.power(G_line, 2)
    G_line_demand = remove_interference_node(G_line_square,G)
    G_final = nx.DiGraph(G_line_demand)
    g = dgl.from_networkx(G_final)
    return g

def shuffle(lst):
    temp_lst = deepcopy(lst)
    m = len(temp_lst)
    while (m):
        m -= 1
        i = randint(0, m)
        temp_lst[m], temp_lst[i] = temp_lst[i], temp_lst[m]
    return temp_lst

from networkx.utils import open_file
@open_file(1, mode="wb")
def write_edgelist(G, path, comments="#", delimiter=" ", data=True, encoding="utf-8"):
    info = "p edge {} {}".format(G.number_of_nodes(), G.number_of_edges())
    info += "\n"
    path.write(info.encode(encoding))
    for line in nx.readwrite.edgelist.generate_edgelist(G, delimiter, data):
        line = "e " + line
        line += "\n"
        path.write(line.encode(encoding))
    return

