import time
from tqdm import tqdm
import networkx as nx
import numpy as np
from data.graph_dataset_color import get_dataset

def envaluate_greedy(data_dir,strategy = 'largest_first',interchange = False, num_color = 0):

    datasets = {
            "vali": get_dataset("vali", data_dir = data_dir, num_nodes_test = 128)
    }

    #strategy= 'largest_first' 'smallest_last'

    C_list = []
    start_time = time.time()
    for i in tqdm(range(datasets['vali'].__len__())):
        g = datasets['vali'][i]
        G = nx.Graph(g.to_networkx())
        d = nx.algorithms.coloring.greedy_color(G, strategy= strategy ,interchange=interchange)

        C = np.max(list(d.values())) + 1  # chromatic number
        C_list.append(C)
    end_time = time.time()

    #C_list = np.array(C_list)
    num_succ = C_list.count(num_color)
    num_total = len(C_list)
    run_time = end_time - start_time
    print("colored in {}".format(run_time))
    #print(np.unique(C_list,return_counts=True))
    #print(strategy,interchange)
    print('successfull rate:', num_succ/num_total)


    return num_succ, num_total, run_time
