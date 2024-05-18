import os
import numpy as np
import networkx as nx
import shutil
from networkx.readwrite.edgelist import read_edgelist
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dgl
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from data.graph_dataset_color import get_dataset
from envs import env_local_color_subspace
import matplotlib.pyplot as plt
import time
import matplotlib.colors as colors
import matplotlib.cm as cmx

def envaluate_RL(data_dir,num_color,max_num_nodes, device, base_model_save_dir, save_graph, space_dim, index = None
                                  ):
    device = torch.device(device)

    print('model:',base_model_save_dir)
    # env

    local_give_up_coef = 0.7
    # actor critic
    num_layers = 3
    hidden_dim = 128
    max_epi_t = 64
    print('T:',max_epi_t)
    # dataset specific
    max_num_nodes = max_num_nodes
    eval_num_samples = 400
    eval_batch_size = 20
    if index is not None:
        num_eval_graphs = len(index)
    else:
        num_eval_graphs = len([
            name
            for name in os.listdir(data_dir)
            if name.endswith('.txt')
        ])
    print('num_graphs:', num_eval_graphs)




    # construct datasets

    datasets = {
        "vali": get_dataset("vali", data_dir=data_dir, num_nodes_test=max_num_nodes, index=index)
    }

    def collate_fn(graphs):
        return dgl.batch(graphs)

    data_loaders = {
        "vali": DataLoader(
            datasets["vali"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=12
        )
    }
    # construct actor critic network
    actor_critic = ActorCritic(
        actor_class=PolicyGraphConvNet,
        critic_class=ValueGraphConvNet,
        max_num_nodes=max_num_nodes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        num_color=num_color
    )
    model_save_dir = os.path.join(base_model_save_dir, str(num_color))
    model_path = os.path.join(model_save_dir, "saved_model")

    actor_critic.load_state_dict(torch.load(model_path, map_location=device))
    actor_critic.to(device)


    env = env_local_color_subspace.MaximumIndependentSetEnv(
        max_epi_t=max_epi_t,
        max_num_nodes=max_num_nodes,
        device=device,
        num_color=num_color,
        local_give_up_coef = local_give_up_coef,
        space_dim = space_dim
    )

    exact_sol_list = []
    succ_colored_list= []
    #test
    def evaluate(mode, actor_critic):
        actor_critic.eval()
        cum_cnt = 0
        cum_suc_count = 0
        exact_succ_index = torch.tensor([])
        for g_di in tqdm(data_loaders[mode]):  # input digraphs
            g_di.set_n_initializer(dgl.init.zero_initializer)
            g_undi = dgl.to_bidirected(g_di, copy_ndata=True) #seperate directed or undirected
            ob = env.register(g_di, num_samples=eval_num_samples)
            while True:
                with torch.no_grad():
                    action = actor_critic.act(ob, g_undi)

                ob, reward, done, info, max_num_in_color = env.step(action)
                if torch.all(done).item():

                    # find the exact sol
                    sol_arr = np.array(info['sol'].cpu()).squeeze()
                    if g_di.batch_size > 1:
                        success_color_count = (sol_arr.max(axis=1) == max_num_nodes).sum()
                        succ_colored = np.apply_along_axis(lambda row: np.where(row == max_num_nodes, 1, 0), axis=1,
                                                           arr=sol_arr).squeeze()
                        for l in succ_colored.max(axis=1).tolist():
                            succ_colored_list.append(l)
                    else:
                        success_color_count = (sol_arr.max() == max_num_nodes).sum()
                        succ_colored = np.apply_along_axis(lambda row: np.where(row == max_num_nodes, 1, 0), axis=0,
                                                           arr=sol_arr).squeeze()
                        l = succ_colored.max().tolist()
                        succ_colored_list.append(l)
                    cum_suc_count += success_color_count

                    final_index = np.argmax(succ_colored, axis=-1)
                    succ_index = np.squeeze(np.where(succ_colored.max(axis=-1) == 1))
                    suc_filename = g_di.ndata['filename'][(succ_index) * max_num_nodes]
                    if suc_filename.ndim > 1:
                        suc_filename = suc_filename.squeeze(dim=1)
                    exact_succ_index = torch.cat((exact_succ_index, suc_filename))
                    ob_arr = np.array(ob.cpu()).squeeze()
                    ob_arr = np.array(np.split(ob_arr, g_di.batch_size))[:, :, :, 0]
                    if g_di.batch_size >1:
                        for i in range(g_di.batch_size):
                            exact_sol = ob_arr[i, :, final_index[i]]
                            exact_sol_list.append(exact_sol)
                    else:
                        exact_sol = ob_arr[:, final_index]
                        exact_sol_list.append(exact_sol)


                    cum_cnt += g_di.batch_size
                    break

        return cum_suc_count, cum_cnt, succ_colored_list, exact_succ_index

    start_time = time.time()
    cum_success_color_count, cum_cnt, succ_colored_list, exact_succ_index= evaluate("vali", actor_critic)
    end_time = time.time()
    run_time = end_time - start_time
    success_ratio = cum_success_color_count / cum_cnt

    print('success ratio {:.4f}'.format(success_ratio))

    save_graph_or_not = save_graph
    save_unsecc = True
    if success_ratio == 0.0:
        save_graph_or_not = False
    if save_graph_or_not:

        vecs = torch.zeros(2 ** space_dim, space_dim, device=device,dtype=int)
        for i in range(2 ** space_dim):
            bin_str = format(i, f'0{space_dim}b')
            vec = torch.tensor([int(b) for b in bin_str])
            vecs[i] = vec
        vecs = vecs.tolist()

        ColorLegend = {}
        for i in range(len(vecs)):
            ColorLegend.update({str(vecs[i]): i})

        suc_dir = os.path.join(data_dir, "subspace_dim{}".format(space_dim),"success")
        unsuc_dir = os.path.join(data_dir,"subspace_dim{}".format(space_dim), "unsuccess")
        if os.path.exists(suc_dir):
            shutil.rmtree(suc_dir)
        if os.path.exists(unsuc_dir):
            shutil.rmtree(unsuc_dir)
        os.makedirs(suc_dir, exist_ok=True)
        os.makedirs(unsuc_dir, exist_ok=True)
        for case in range(num_eval_graphs):
            if succ_colored_list[case] == 1:#succ_local_colored
                plot_save_path = os.path.join(suc_dir, "{:06d}.png".format(case))
            else:
                if save_unsecc:
                    plot_save_path = os.path.join(unsuc_dir, "{:06d}.png".format(case))
                else:
                    continue
            read_dir = os.path.join(data_dir,"{:06d}.txt".format(case))
            g = dgl.from_networkx(nx.relabel.convert_node_labels_to_integers(read_edgelist(read_dir, create_using=nx.DiGraph),
                                                                         first_label=0))
            g.add_nodes(max_num_nodes - g.number_of_nodes())
            # 找到孤立点
            isolated_nodes = list(nx.isolates(g.to_networkx()))

            # 从图中删除孤立点
            g.remove_nodes(isolated_nodes)

            # 从 exact_sol_list[case] 中删除对应的节点值
            exact_sol_list[case] = [exact_sol_list[case][i] for i in range(len(exact_sol_list[case])) if
                                    i not in isolated_nodes]

            #pos = nx.circular_layout(g.to_networkx())
            pos = nx.kamada_kawai_layout(g.to_networkx())
            jet = plt.get_cmap('jet')
            cNorm = colors.Normalize(vmin=0, vmax=max(exact_sol_list[case]))
            scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

            # Using a figure to use it as a parameter when calling nx.draw_networkx
            f = plt.figure(1)
            ax = f.add_subplot(1, 1, 1)
            for label in ColorLegend:
                ax.plot([0], [0], color=scalarMap.to_rgba(ColorLegend[label]), label=label)

            nx.draw_networkx(g.to_networkx(), pos, cmap=jet, vmin=0, vmax=max(exact_sol_list[case]),
                             node_color=exact_sol_list[case], with_labels=True, ax=ax)

            plt.axis('off')
            f.set_facecolor('w')
            plt.legend()
            f.tight_layout()
            plt.savefig(plot_save_path,dpi=300)
            plt.close('all')

            # read_dir = os.path.join(data_dir, "{:06d}.txt".format(case))
            # g = dgl.from_networkx(nx.relabel.convert_node_labels_to_integers(
            #     nx.read_edgelist(read_dir, create_using=nx.DiGraph), first_label=0))
            # g.add_nodes(max_num_nodes - g.number_of_nodes())
            # pos = nx.circular_layout(g.to_networkx())
            # jet = plt.get_cmap('jet')
            # cNorm = colors.Normalize(vmin=0, vmax=max(exact_sol_list[case]))
            # scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)
            #
            # # Using a figure to use it as a parameter when calling nx.draw_networkx
            # f = plt.figure(1)
            # ax = f.add_subplot(1, 2, 1)  # Subplot for original graph
            # for label in ColorLegend:
            #     ax.plot([0], [0], color=scalarMap.to_rgba(ColorLegend[label]), label=label)
            #
            # nx.draw_networkx(g.to_networkx(), pos, cmap=jet, vmin=0, vmax=max(exact_sol_list[case]),
            #                  node_color=exact_sol_list[case], with_labels=True, ax=ax)
            #
            # plt.axis('off')
            # f.set_facecolor('w')
            # plt.legend()
            #
            # # Draw complement graph
            # complement_g = nx.complement(g.to_networkx())
            # complement_pos = nx.spring_layout(complement_g)  # You can choose a different layout if you prefer
            # ax = f.add_subplot(1, 2, 2)  # Subplot for complement graph
            # nx.draw_networkx(complement_g, complement_pos, node_color='lightgray', with_labels=True, ax=ax)
            # plt.axis('off')
            #
            # f.tight_layout()
            # plt.savefig(plot_save_path, dpi=300)
            # plt.close('all')


    return cum_success_color_count, cum_cnt, run_time, exact_succ_index.int().tolist()