import os
import shutil
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dgl
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from data.graph_dataset_color import get_dataset
from envs import env_local_color_simo
import matplotlib.pyplot as plt
import time


def envaluate_RL_local_same_model(data_dir,num_color,max_num_nodes, device, training_device, base_model_save_dir, save_graph
                                  ,clean2,index = None):
    device = torch.device(device)
    training_device = torch.device(training_device)

    # env
    local_give_up_coef = 0.7
    # actor critic
    num_layers = 3
    hidden_dim = 128
    max_epi_t = 128
    print('T:',max_epi_t)
    # dataset specific
    max_num_nodes = max_num_nodes
    #eval_num_samples_total = 100*64

    if index is not None:
        num_eval_graphs = len(index)
    else:
        num_eval_graphs = len([
            name
            for name in os.listdir(data_dir)
            if name.endswith('.txt')
        ])
    print('num_graphs:', num_eval_graphs)
    #eval_num_samples = 20
    #eval_num_samples = int(eval_num_samples_total/num_eval_graphs)
    eval_num_samples = 200
    eval_batch_size = 128


    # construct datasets

    datasets = {
        "vali": get_dataset("vali", data_dir = data_dir, num_nodes_test = max_num_nodes, index = index)
    }

    def collate_fn(graphs):
        return dgl.batch(graphs)

    data_loaders = {
        "vali": DataLoader(
            datasets["vali"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=4,
            pin_memory=True
        )
    }
    actor_critic = ActorCritic(
        actor_class=PolicyGraphConvNet,
        critic_class=ValueGraphConvNet,
        max_num_nodes=max_num_nodes,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        device=device,
        num_color=num_color
    )
    model_save_dir = base_model_save_dir
    model_path = os.path.join(model_save_dir, "saved_model")

    actor_critic.load_state_dict(torch.load(model_path, map_location={'{}'.format(training_device):'{}'.format(device)}))
    actor_critic.to(device)


    env = env_local_color_simo.MaximumIndependentSetEnv(
        max_epi_t=max_epi_t,
        max_num_nodes=max_num_nodes,
        device=device,
        num_color=num_color,
        local_give_up_coef = local_give_up_coef,
        clean2 = clean2
    )

    exact_sol_list = []
    succ_local_colored_list = []
    #test
    def evaluate(mode, actor_critic,index):
        actor_critic.eval()
        cum_cnt = 0
        cum_success_color_count = 0
        cum_success_local_color_count = 0
        for g_di in tqdm(data_loaders[mode]):  # input digraphs
            g_di.set_n_initializer(dgl.init.zero_initializer)
            g_undi = dgl.to_bidirected(g_di, copy_ndata=True) #seperate directed or undirected
            g_undi.set_n_initializer(dgl.init.zero_initializer)
            ob = env.register(g_di, num_samples=eval_num_samples)
            while True:
                with torch.no_grad():
                    action = actor_critic.act(ob, g_undi)

                ob, reward, done, info, max_num_in_color = env.step(action)
                if torch.all(done).item():
                    max_num_in_color = np.array(max_num_in_color.cpu())
                    #print(max_num_in_color)
                    succ_local_colored = np.apply_along_axis(lambda row: np.where(row+1 <= num_color - clean2, 1, 0), axis=1,
                                                             arr=max_num_in_color).squeeze()
                    sol_arr = np.array(info['sol'].cpu()).squeeze()
                    try:
                        succ_colored = np.apply_along_axis(lambda row: np.where(row == max_num_nodes, 1, 0), axis=1,
                                                       arr=sol_arr).squeeze()
                    except:
                        succ_colored = np.apply_along_axis(lambda row: np.where(row == max_num_nodes, 1, 0), axis=0,
                                                       arr=sol_arr).squeeze()
                    succ_local_colored[np.where(succ_colored == 0)] = 0
                    try:
                        success_color_count = succ_colored.max(axis=1).sum()
                    except:
                        success_color_count = succ_colored.max().sum()
                    try:
                        success_local_color_count = succ_local_colored.max(axis=1).sum()
                        suc_local_index = np.where(succ_local_colored.max(axis=1)==1)[0]
                        if index is None:
                            exact_index = suc_local_index
                        else:
                            exact_index = (np.array(index)[suc_local_index]).tolist()
                    except:
                        success_local_color_count = succ_local_colored.max().sum()
                    print('success_local_color_count',success_local_color_count)


                    cum_cnt += g_di.batch_size
                    cum_success_color_count += success_color_count
                    cum_success_local_color_count += success_local_color_count

                    #solution:
                    for l in succ_local_colored.max(axis=1).tolist():
                        succ_local_colored_list.append(l)

                    final = succ_colored.copy()
                    final[np.where(succ_local_colored == 1)] = 2
                    final_index = np.argmax(final, axis=1)
                    ob_arr = np.array(ob.cpu()).squeeze()
                    ob_arr = np.array(np.split(ob_arr, g_di.batch_size))[:, :, :, 0]

                    for i in range(g_di.batch_size):
                        exact_sol = ob_arr[i, :, final_index[i]]
                        exact_sol_list.append(exact_sol)

                    break

        return cum_success_color_count, cum_success_local_color_count, cum_cnt, exact_index

    start_time = time.time()
    cum_success_color_count, cum_success_local_color_count, cum_cnt, exact_index = evaluate("vali", actor_critic,index)
    end_time = time.time()
    run_time = end_time - start_time
    success_ratio = cum_success_color_count / cum_cnt
    succes_local_ratio = cum_success_local_color_count / cum_cnt

    print('success ratio {:.4f}'.format(success_ratio))
    print('success local ratio {:.4f}'.format(succes_local_ratio))

    save_graph_or_not = save_graph
    save_unsecc = False
    if succes_local_ratio == 0.0:
        save_graph_or_not = False
    if save_graph_or_not:
        if index is None:
            index = list(range(num_eval_graphs))
        suc_dir = os.path.join(data_dir, "local_color_simo_clean{}".format(clean2),"success")
        unsuc_dir = os.path.join(data_dir,"local_color_simo_clean{}".format(clean2), "unsuccess")
        if os.path.exists(suc_dir):
            shutil.rmtree(suc_dir)
        if os.path.exists(unsuc_dir):
            shutil.rmtree(unsuc_dir)
        os.makedirs(suc_dir, exist_ok=True)
        os.makedirs(unsuc_dir, exist_ok=True)
        for case in range(num_eval_graphs):
            if succ_local_colored_list[case] == 1:#succ_local_colored
                plot_save_path = os.path.join(suc_dir, "{:06d}.jpg".format(index[case]))
            else:
                if save_unsecc:
                    plot_save_path = os.path.join(unsuc_dir, "{:06d}.jpg".format(index[case]))
                else:
                    continue
            read_dir = os.path.join(data_dir,"{:06d}.txt".format(index[case]))
            g = dgl.from_networkx(nx.relabel.convert_node_labels_to_integers(nx.readwrite.edgelist.read_edgelist(read_dir, create_using=nx.DiGraph),
                                                                         first_label=0))
            g.add_nodes(max_num_nodes - g.number_of_nodes())
            nx.draw_networkx(g.to_networkx(), pos=nx.circular_layout(g.to_networkx()), with_labels=True,
                             font_weight='bold', node_color=exact_sol_list[case])
            plt.savefig(plot_save_path)
            plt.close('all')



    return cum_success_color_count, cum_success_local_color_count, cum_cnt, run_time, exact_index