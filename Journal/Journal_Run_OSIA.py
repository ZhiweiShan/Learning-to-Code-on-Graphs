import os
from envaluate.Journal_Envaluate_local_color import envaluate_RL
import pickle
import sys
# OSIA-(1,num_color-clean2), dof=1/(num_color-clean2)
method = 'OSIA'
data_dir_base = 'data/new_data_di_ER_6_0.3_1000_test_chromatic'

b = 1
num_color = 3
num_color_max = 10
clean2 = b
orig_max_num_nodes = 6
save_graph_picture = False
try:
    data_dir_base = sys.argv[1]
    orig_max_num_nodes = int(sys.argv[2])
except:
    pass
max_num_nodes = orig_max_num_nodes*b
data_dir_di = os.path.join(data_dir_base,'directed')
base_model_save_dir = 'model_save'
result_dir = os.path.join(data_dir_base,'OSIA.pickle')
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
#first try 1/2, drop the cases succ, try 1/3 for the rest cases
while num_color <= num_color_max:
    print('****Start OSIA {}-{}****'.format(b,num_color-clean2))
    cum_success_color_count, cum_success_local_color_count, cum_cnt, run_time, success_index = envaluate_RL(method,data_dir_di,
               num_color,max_num_nodes, device= 0, base_model_save_dir = base_model_save_dir,
               save_graph = save_graph_picture, clean2 = clean2, index= remain_idx)

    print('time:',run_time)
    # find unsuccess_index by removing success_index from remain_idx
    remain_idx = [x for x in remain_idx if x not in success_index]
    rate.append(b/(num_color-clean2))
    cum_success_count.append(cum_success_local_color_count)
    achieve_dof_index['{}-{}'.format(b,num_color-clean2)] = success_index

    #print('\n success graph:', success_index)
    if len(remain_idx) == 0:
        print('all finished')
        break
    num_color += 1


with open(result_dir, 'wb') as f:
    pickle.dump(achieve_dof_index, f)
rate = [float('{:.2f}'.format(i)) for i in rate]

print('OSIA')
print('DoF: {}'.format(rate))
print('success: {}'.format(cum_success_count))