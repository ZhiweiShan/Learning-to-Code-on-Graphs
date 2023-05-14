import matlab.engine
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from generate_data.topology2conflict import generate_havel_hakimi_graph, shuffle, t2c_directed, t2c_undirected, generate_random_bigraph, generate_random_triangle_graph, remove_some_direction
import dgl
eng = matlab.engine.start_matlab()

def generate_network(num_t,num_r,num_antenna,p):
    # p: percent of interference
    # 1-p: percent of ignored connection
# range = 1000, maxDist = 65; minDist = 2; numAntenna = 1; c = 3e8 speed of light; freq = 2.4e9; % in Hz
    H,G = eng.generateNetwork(num_t, num_r, num_antenna, nargout=2)
    G = np.array(G)
    G[G <= np.percentile(G, 100 * (1 - p))] = 0
    G[G > np.percentile(G, 100 * (1 - p))] = 2
    row, col = np.diag_indices_from(G)
    G[row, col] = 1
    G_nx = nx.Graph()
    demand_edges = np.array(np.where(G == 1)).transpose()
    demand_edges[:, 1] = demand_edges[:, 1] + num_t
    inter_edges = np.array(np.where(G == 2)).transpose()
    inter_edges[:, 1] = inter_edges[:, 1] + num_t
    inter_edges = tuple(map(tuple, inter_edges))
    G_nx.add_edges_from(demand_edges)
    nx.set_edge_attributes(G_nx, 1, 'demand')
    G_nx.add_edges_from(inter_edges)
    diction = {i: 2 for i in inter_edges}
    nx.set_edge_attributes(G_nx, diction, 'demand')
    return G_nx


if __name__ == "__main__":
    G = generate_network(5, 5, 1, 0.7)
    top = nx.bipartite.sets(G)[0]
    pos = nx.bipartite_layout(G, top)
    nx.draw_networkx(G, pos=pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos=pos)
    plt.show()

    G_conflict = t2c_undirected(G)
    nx.draw(G_conflict.to_networkx(), with_labels=True, font_weight='bold')
    plt.show()