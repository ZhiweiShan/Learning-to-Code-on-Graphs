import os
import argparse
from time import time
import torch
from torch.utils.data import DataLoader
import dgl
from ppo.framework import ProxPolicyOptimFramework
from ppo.actor_critic import ActorCritic

from ppo.graph_net import PolicyGraphConvNet, ValueGraphConvNet
from ppo.storage_gpu import RolloutStorage
from data.graph_dataset_color import get_dataset

from envs import env_color
import matplotlib.pyplot as plt

cpu_num = 1
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)


parser = argparse.ArgumentParser()
parser.add_argument(
    "--num-color",
    help="number of color to use",
    type=str
    )
parser.add_argument(
    "--data-dir", 
    help="directory of training datasets",
    type=str
    )
parser.add_argument(
    "--model-save-dir",
    help="directory to store trained model",
    type=str
    )
parser.add_argument(
    "--device",
    help="id of gpu device to use",
    type=int
    )
parser.add_argument(
    "--num-nodes",
    help="max number of nodes of training graphs",
    type=int
    )
args = parser.parse_args()
num_color = int(args.num_color)
device = torch.device(args.device)
base_data_dir = os.path.join(args.data_dir)
base_model_save_dir = os.path.join(args.model_save_dir)
num_nodes_train = args.num_nodes
# env
hamming_reward_coef = 0

# actor critic
num_layers = 3
hidden_dim = 128

# optimization
init_lr = 1e-3
max_epi_t = 64
max_rollout_t = max_epi_t
max_update_t = 10000

# ppo
gamma = 1.0
clip_value = 0.2
optim_num_samples = 4
critic_loss_coef = 0.5
reg_coef = 0.1
max_grad_norm = 0.5

# logging
log_freq = 10



# main
rollout_batch_size = 32
optim_batch_size = 16
init_anneal_ratio = 1.0
max_anneal_t = - 1
anneal_base = 0.
train_num_samples = 2
eval_num_samples = 10

# initial values
best_vali_sol = -1e5



mode="vali"


# define evaluate function
def evaluate(mode, actor_critic):
    actor_critic.eval()
    cum_cnt = 0.00001
    cum_eval_sol = 0.00001
    for g in data_loaders[mode]: #mode
        g.set_n_initializer(dgl.init.zero_initializer)
        ob = env.register(g, num_samples = eval_num_samples)
        while True:
            with torch.no_grad():
                action = actor_critic.act(ob, g)

            ob, reward, done, info, in_color_reward = env.step(action)
            if torch.all(done).item():
                cum_eval_sol += info['sol'].max(dim = 1)[0].sum().cpu()
                cum_cnt += g.batch_size
                break

    actor_critic.train()
    avg_eval_sol = cum_eval_sol / cum_cnt
    return avg_eval_sol

sol_list = []
vali_list = []
in_color_reward_list = []
list_index = 0

model_save_dir = os.path.join(base_model_save_dir, str(num_color))
os.makedirs(model_save_dir, exist_ok=True)
print("model_save_dir:",model_save_dir)


# construct datasets

max_num_nodes = num_nodes_train
datasets = {
    "train": get_dataset("train", data_dir = base_data_dir, num_nodes_train = num_nodes_train),
}

# construct data loaders
def collate_fn(graphs):
    return dgl.batch(graphs)

data_loaders = {
    "train": DataLoader(
        datasets["train"],
        batch_size=rollout_batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        drop_last=True,
        pin_memory = True
    )
}


rollout = RolloutStorage(
    max_t=max_rollout_t,
    batch_size=rollout_batch_size,
    num_samples=train_num_samples,
    device=device
)

# construct actor critic network
actor_critic = ActorCritic(
    actor_class=PolicyGraphConvNet,
    critic_class=ValueGraphConvNet,
    max_num_nodes=max_num_nodes,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    device=device,
    num_color = num_color
)

# construct PPO framework
framework = ProxPolicyOptimFramework(
    actor_critic=actor_critic,
    init_lr=init_lr,
    clip_value=clip_value,
    optim_num_samples=optim_num_samples,
    optim_batch_size=optim_batch_size,
    critic_loss_coef=critic_loss_coef,
    reg_coef=reg_coef,
    max_grad_norm=max_grad_norm,
    device=device
)
# construct environment
env = env_color.MaximumIndependentSetEnv(
    max_epi_t=max_epi_t,
    max_num_nodes=max_num_nodes,
    hamming_reward_coef=hamming_reward_coef,
    device=device,
    num_color = num_color
)

start_time = time()

for update_t in range(max_update_t):
    if update_t == 0 or torch.all(done).item():
        try:
            g = next(train_data_iter)
        except:
            train_data_iter = iter(data_loaders["train"])
            g = next(train_data_iter)

        g.set_n_initializer(dgl.init.zero_initializer)
        ob = env.register(g, num_samples = train_num_samples)
        rollout.insert_ob_and_g(ob, g)

    for step_t in range(max_rollout_t):
        # get action and value prediction

        with torch.no_grad():
            (action,
            action_log_prob,
            value_pred,
            ) = actor_critic.act_and_crit(ob, g)

        # step environments
        ob, reward, done, info, in_color_reward = env.step(action)

        # insert to rollout
        rollout.insert_tensors(
            ob,
            action,
            action_log_prob,
            value_pred,
            reward,
            done
            )

        if torch.all(done).item():
            avg_sol = info['sol'].max(dim = 1)[0].mean().cpu()
            break

    # compute gamma-decayed returns and corresponding advantages
    rollout.compute_rets_and_advantages(gamma)

    # update actor critic model with ppo
    actor_loss, critic_loss, entropy_loss = framework.update(rollout)
    sol_list.append(avg_sol)
    in_color_reward_list.append(in_color_reward)
    # log stats
    if (update_t + 1) % log_freq == 0:
        print("update_t: {:05d}".format(update_t + 1))
        print("train stats...{}-color".format(num_color))
        print("average colored nodes: {:.4f} ".format(avg_sol))
list_index += 1
model_save_dir = os.path.join(base_model_save_dir, str(num_color))
os.makedirs(model_save_dir, exist_ok=True)
model_path = os.path.join(model_save_dir, "saved_model")
torch.save(actor_critic.state_dict(), model_path)
end_time = time()
print("training in {:.3f}".format(end_time - start_time))
print(model_save_dir)

plt.plot(sol_list)
plt.show()
#plt.plot(in_color_reward_list)