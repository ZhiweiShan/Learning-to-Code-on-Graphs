import os
from envaluate.Journal_Envaluate_matrix_rank_reduction import envaluate_RL
import pickle
import sys
# SSIA-(b,space_dim), dof=b/(space_dim)
method = 'SSIA'
data_dir_base = '../data/new_data_di_ER_6_0.3_1000_test_chromatic'
b = 1
space_dim = 2
space_dim_max = 12
orig_max_num_nodes = 6


save_graph_picture = False
try:
    data_dir_base = sys.argv[1]
    orig_max_num_nodes = int(sys.argv[2])
except:
    pass
data_dir_di = os.path.join(data_dir_base,'directed')
max_num_nodes = b * orig_max_num_nodes
base_model_save_dir = '../model_save'

print('b',b)
result_dir = os.path.join(data_dir_base,'SSIA.pickle')

num_eval_graphs = len([
            name
            for name in os.listdir(data_dir_di)
            if name.endswith('.txt')
        ])
all_graph_idx = list(range(num_eval_graphs))
remain_idx = all_graph_idx

cum_success_count = []
rate = []
success_index = None
achieve_dof_index = {}

while space_dim <= space_dim_max:
    num_vecs = min(2 ** space_dim - 1, 8)
    print('****Start SSIA{}-{}****'.format(b,space_dim))
    cum_success_color_count, cum_cnt, run_time, success_index  = envaluate_RL(data_dir_di,
               num_vecs,max_num_nodes, device= 0, base_model_save_dir = base_model_save_dir,
               save_graph = save_graph_picture, space_dim = space_dim, index= remain_idx)

    print('time:',run_time)
    # find unsuccess_index by removing success_index from remain_idx
    remain_idx = [x for x in remain_idx if x not in success_index]
    rate.append(b/(space_dim))
    cum_success_count.append(len(success_index))
    achieve_dof_index['{}-{}'.format(b,space_dim)] = success_index

    #print('\n success graph:', success_index)
    if len(remain_idx) == 0:
        print('all finished')
        break
    space_dim += 1


with open(result_dir, 'wb') as f:
    pickle.dump(achieve_dof_index, f)
rate = [float('{:.2f}'.format(i)) for i in rate]
print('SSIA')
print('DoF: {}'.format(rate))
print('success: {}'.format(cum_success_count))