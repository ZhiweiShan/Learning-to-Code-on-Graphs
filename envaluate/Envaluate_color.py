import os
import numpy as np
import networkx as nx
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import dgl
from ppo.actor_critic import ActorCritic
from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from data.graph_dataset_color import get_dataset
from envs import env_color
from dgl.data.utils import load_graphs
import matplotlib.pyplot as plt
import time


def envaluate_RL(data_dir,num_color,max_num_nodes, device, training_device, base_model_save_dir):
    device = torch.device(device)
    training_device = torch.device(training_device)


    print('model:',base_model_save_dir)
    # env
    hamming_reward_coef = 0

    # actor critic
    num_layers = 3
    hidden_dim = 128
    max_epi_t = 32
    # dataset specific
    max_num_nodes = max_num_nodes
    eval_num_samples = 20
    eval_batch_size = 2048
    num_eval_graphs = len([
                    name
                    for name in os.listdir(data_dir)
                    if name.endswith('.txt')
                    ])
    print('num_graphs:', num_eval_graphs)


    exact_sol_list = []

    # construct datasets
    datasets = {
        "vali": get_dataset("vali", data_dir = data_dir, num_nodes_test = max_num_nodes)
    }

    def collate_fn(graphs):
        return dgl.batch(graphs)

    data_loaders = {
        "vali": DataLoader(
            datasets["vali"],
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0
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

    actor_critic.load_state_dict(torch.load(model_path, map_location={'{}'.format(training_device):'{}'.format(device)}))
    actor_critic.to(device)


    env = env_color.MaximumIndependentSetEnv(
        max_epi_t=max_epi_t,
        max_num_nodes=max_num_nodes,
        hamming_reward_coef=hamming_reward_coef,
        device=device,
        num_color=num_color
    )

    #test
    def evaluate(mode, actor_critic):
        actor_critic.eval()
        cum_cnt = 0
        cum_suc_count = 0
        for g in tqdm(data_loaders[mode]):  # mode
            g.set_n_initializer(dgl.init.zero_initializer)
            ob = env.register(g, num_samples=eval_num_samples)
            while True:
                with torch.no_grad():
                    action = actor_critic.act(ob, g)

                ob, reward, done, info, in_color_reward = env.step(action)
                if torch.all(done).item():
                    cum_cnt += g.batch_size
                    #find the exact sol
                    sol_arr = np.array(info['sol'].cpu()).squeeze()

                    try:
                        cum_suc_count += (sol_arr.max(axis=1) == max_num_nodes).sum()
                    except:
                        cum_suc_count += (sol_arr.max() == max_num_nodes).sum()
                        print('\nmax colored nodes',sol_arr.max())

                    break


        return cum_suc_count, cum_cnt

    start_time = time.time()
    cum_suc_count , cum_cnt = evaluate("vali", actor_critic)
    end_time = time.time()
    run_time = end_time - start_time
    success_ratio = cum_suc_count / cum_cnt
    print('colored in {}'.format(run_time))
    print('success ratio {:.4f}'.format(success_ratio))



    save_graph_or_not = False
    if save_graph_or_not:
        suc_dir = os.path.join(data_dir, "success")
        unsuc_dir = os.path.join(data_dir, "unsuccess")
        os.makedirs(suc_dir, exist_ok=True)
        os.makedirs(unsuc_dir, exist_ok=True)
        for case in range(num_eval_graphs):
            if (np.max(exact_sol_list[0],axis = 1) == num_color)[case]:
                plot_save_path = os.path.join(suc_dir, "{:06d}.jpg".format(case))
            else:
                plot_save_path = os.path.join(unsuc_dir, "{:06d}.jpg".format(case))
            read_dir = os.path.join(data_dir,"{:06d}.METIS".format(case))
            g = load_graphs(read_dir)[0][0]
            nx.draw_networkx(g.to_networkx(),pos = nx.circular_layout(g.to_networkx()), with_labels=True, font_weight='bold',node_color =exact_sol_list[0][case] )
            plt.savefig(plot_save_path)
            plt.close('all')


    return cum_suc_count , cum_cnt, run_time