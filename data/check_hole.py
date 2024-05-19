import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import os
from networkx.readwrite.edgelist import read_edgelist
matplotlib.use('TkAgg')

# 创建一个示例图
# G = nx.Graph()
# G.add_edges_from([(1, 2), (2, 3), (4, 1), (2, 5), (3, 5), (4, 5), (3,6), (4,6), (5,6)])
# pos = nx.kamada_kawai_layout(G)
# nx.draw_networkx(G, pos,  with_labels=True)
# plt.pause(0.1)
# plt.show()
# print(list(nx.cycle_basis(G)))
#
# chordless_cycles = sorted(list(nx.chordless_cycles(G)))
# print('chordless_cycles:',chordless_cycles)
# print([len(cycle) for cycle in chordless_cycles])

def check_odd_hole(G):
    chordless_cycles = sorted(list(nx.chordless_cycles(G)))
    len_list = [len(cycle) for cycle in chordless_cycles]
    odd_numbers_greater_than_5 = [num for num in len_list if num >= 5 and num % 2 != 0]
    if len(odd_numbers_greater_than_5) == 0:
        return False
    if len(odd_numbers_greater_than_5) >= 1:
        return True
def load_G_di(idx):
    data_dir = '../data/data_di_network_8810.4_1000_test/directed/chromaticed/4'
    g_path = os.path.join(data_dir, "{:06d}.txt".format(idx))
    G = nx.relabel.convert_node_labels_to_integers(read_edgelist(g_path, create_using=nx.DiGraph), first_label=0)
    return G
def load_G_undi(idx):
    data_dir = '../data/data_di_network_8810.5_1000_test/directed/chromaticed/4'
    g_path = os.path.join(data_dir, "{:06d}.txt".format(idx))
    G = nx.relabel.convert_node_labels_to_integers(read_edgelist(g_path, create_using=nx.Graph), first_label=0)
    return G
if __name__ == '__main__':
    # data_dir = 'data_di_network_8810.4_1000_test/directed/chromaticed/4'
    data_dir = '../data/data_di_network_8810.4_1000_test/directed/chromaticed/4'
    num_graphs = len([
        name
        for name in os.listdir(data_dir)
        if name.endswith('.txt')])
    odd_hole_list = []
    have_odd_hole_list = []
    not_have_odd_hole_list = []
    for idx in range(num_graphs):
        g_path = os.path.join(data_dir,"{:06d}.txt".format(idx))

        G = nx.relabel.convert_node_labels_to_integers(read_edgelist(g_path, create_using=nx.Graph), first_label=0)
        have_odd_hole = check_odd_hole(G)
        if have_odd_hole:
            have_odd_hole_list.append(idx)
        else:
            not_have_odd_hole_list.append(idx)

    unsucc_dir = '../data/data_di_network_8810.4_1000_test/directed/chromaticed/4/subspace_dim3/unsuccess'
    unsucc_list = [
        name
        for name in os.listdir(unsucc_dir)
        if name.endswith('.png')]

    unsucc_numbers = [int(filename.lstrip('0').split('.')[0]) if filename.lstrip('0').split('.')[0] != '' else 0 for filename in unsucc_list]

    # 检查unsucc_numbers中的数字是否在not_have_odd_hole_list中
    unsucc_but_have_odd_hole = [num for num in unsucc_numbers if num not in not_have_odd_hole_list]
    not_have_odd_hole_but_succ = [num for num in not_have_odd_hole_list if num not in unsucc_numbers]

    print("unsucc but have odd hole:", unsucc_but_have_odd_hole)
    print("not have odd hole but succ:", not_have_odd_hole_but_succ)

    better_than_O2O = [22,46,57,61,62,71,93,103,111,156,157,159,166,173,174,204,206,212,217,230,261,262,267,276,295,302,347,357,397,402,422,458,471,
                       482,498,529,535,550,563,566,567,568,585,]
    better_than_O2O_but_not_have_odd_hole = [num for num in better_than_O2O if num in not_have_odd_hole_list]
    # classic triangle
    # 46,57,103,174,204,206,212,230,261,262,267,276,357,397,402,471,482,498,529,563,566,567,
    # new structure
    # 422,535,585