import os.path
from tqdm import tqdm
import networkx as nx


from networkx.utils import open_file
@open_file(1, mode="wb")
def write_edgelist(G, path, comments="#", delimiter=" ", data=True, encoding="utf-8"):
    for line in nx.readwrite.edgelist.generate_edgelist(G, delimiter, data):
        line += "\n"
        path.write(line.encode(encoding))
    return


def node_splitting(G,spl_num):
    G_old = G
    G_new = nx.DiGraph()
    # add a clique for each node
    for node in G_old.nodes():
        node_lable_list = []
        for i in range(1,spl_num+1):
            #G_new.add_node((node,i))
            node_lable_list.append((node,i))
        G1 = nx.complete_graph(spl_num, nx.DiGraph())
        mapping = dict(zip(G1, node_lable_list))
        G1 = nx.relabel_nodes(G1, mapping)
        G_new.add_nodes_from(G1)
        G_new.add_edges_from(G1.edges)
    # add edges according to the edges of old graph
    for (u,v) in G_old.edges():
        edge_list = []
        for i in range(1, spl_num + 1):
            for j in range(1, spl_num + 1):
                edge_list.append(((u,i),(v,j)))
        G_new.add_edges_from(edge_list)
    return G_new

#Data set:
num_color =4
num_split = 2
old_dir = '../data/data_di_bipar_550.40.4_5000_test/directed/chromaticed/{}'.format(num_color)
new_dir = '../data/data_di_bipar_550.40.4_5000_test/directed/chromaticed/{}_split_{}'.format(num_color,num_split)
os.makedirs(new_dir, exist_ok = True)
graph_paths = os.listdir(old_dir)
for graph_path in tqdm(graph_paths):
    if graph_path.endswith(".txt"):
        graph_path_full = os.path.join(old_dir, graph_path)
        G = nx.readwrite.edgelist.read_edgelist(graph_path_full, create_using=nx.DiGraph)
        G_new = node_splitting(G,num_split)
        G_new = nx.convert_node_labels_to_integers(G_new, first_label=1)
        save_path = os.path.join(new_dir,graph_path)
        write_edgelist(G_new, path=save_path, data=False)

