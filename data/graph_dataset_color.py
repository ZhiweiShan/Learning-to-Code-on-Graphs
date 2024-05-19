import os
import random
import math
import networkx as nx
import dgl
import torch
from dgl import DGLGraph
from torch.utils.data import Dataset
import re
from networkx.readwrite.edgelist import read_edgelist



def generate_directed_cycle2(n, p):
    G = dgl.DGLGraph()
    G.add_nodes(n)
    for i in range(n - 1):
        G.add_edge(i, i + 1)
    G.add_edge(n - 1, 0)
    return G


def generate_er_graph_from_networkx(n, p):  # the same as generate_er_graph below

    G_nx = nx.generators.random_graphs.erdos_renyi_graph(n, p)
    G_dgl = DGLGraph(G_nx)

    return G_dgl


def generate_er_graph(n, p):
    G = DGLGraph()
    G.add_nodes(n)
    random.seed(0)
    w = -1
    lp = math.log(1.0 - p)

    # Nodes in graph are from 0,n-1 (start with v as the first node index).
    v = 1
    edges_list = []
    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        while w >= v and v < n:
            w = w - v
            v = v + 1
        if v < n:
            edges_list.extend([(v, w), (w, v)])

    G.add_edges(*zip(*edges_list))

    return G


class GraphDataset(Dataset):
    def __init__(
            self,
            data_dir=None,
            generate_fn=None,
            num_nodes=None,
            index = None
    ):
        self.data_dir = data_dir
        self.generate_fn = generate_fn
        self.num_nodes = num_nodes
        self.index = index

        if data_dir is not None:
            if self.index is not None:
                self.num_graphs = len(self.index)
            else:
                self.num_graphs = len([
                    name
                    for name in os.listdir(data_dir)
                    if name.endswith('.txt')

                ])
        elif generate_fn is not None:
            self.num_graphs = 5000  # 5000 # sufficiently large number for moving average
        else:
            assert False

    def __getitem__(self, idx):
        if self.generate_fn is None:
            if self.index is not None:
                g_path = os.path.join(
                    self.data_dir,
                    "{:06d}.txt".format(self.index[idx])
                )
                feature = {'filename': self.index[idx] * torch.ones((self.num_nodes, 1))}
            else:
                g_path = os.path.join(
                    self.data_dir,
                    "{:06d}.txt".format(idx)
                )
                feature = {'filename': idx * torch.ones((self.num_nodes, 1))}

            g = dgl.from_networkx(nx.relabel.convert_node_labels_to_integers(read_edgelist(g_path,create_using=nx.DiGraph),first_label=0))
            try:
                g.add_nodes(self.num_nodes - g.number_of_nodes())
            except:
                raise ValueError('need to increase the number of nodes')
            g.ndata.update(feature)
        else:
            g = self.generate_fn()

        return g

    def __len__(self):
        return self.num_graphs


def generate_directed_er_graph(n, p, directed=False):
    G = DGLGraph()
    G.add_nodes(n)

    w = -1
    lp = math.log(1.0 - p)
    edges_list = []
    v = 0
    while v < n:
        lr = math.log(1.0 - random.random())
        w = w + 1 + int(lr / lp)
        if v == w:  # avoid self loops
            w = w + 1
        while v < n <= w:
            w = w - n
            v = v + 1
            if v == w:  # avoid self loops
                w = w + 1
        if v < n:
            if directed:
                edges_list.extend([(v, w)])
            else:
                edges_list.extend([(v, w), (w, v)])

    edges_list = list(set(edges_list))
    G.add_edges(*zip(*edges_list))
    return G


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


def check_chromatic(g, num_color):
    G_undi = nx.DiGraph(g.to_networkx())
    G_undi.remove_nodes_from(list(nx.isolates(G_undi)))
    G_undi = nx.convert_node_labels_to_integers(G_undi, first_label=1)

    os.makedirs("training_data_undi", exist_ok=True)
    training_g_path_undi = os.path.join("training_data_undi", "{:06d}.txt".format(1))
    write_edgelist(G_undi, path=training_g_path_undi, data=False)

    main = "../exactcolors-master/color {}".format(training_g_path_undi)
    f = os.popen(main)
    data = f.readlines()
    # print(data)
    f.close()
    str_match = list(filter(lambda x: "Opt Colors" in x, data))
    C = int(re.findall(r"\d+", str_match[0])[0])
    os.remove(training_g_path_undi)
    if C == num_color:
        return True
    else:
        return False


def get_dataset(mode, num_nodes_train=None, data_dir=None, num_nodes_test=None, index = None):
    if mode == "train":
        return GraphDataset(data_dir=data_dir, num_nodes=num_nodes_train)
    else:
        return GraphDataset(data_dir=data_dir, num_nodes=num_nodes_test, index = index)