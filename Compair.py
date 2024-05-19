import os
from envaluate.Envaluate_greedy_based import envaluate_greedy
from envaluate.Envaluate_color import envaluate_RL
from envaluate.Tabu import Tabu

num_color = 3

data_dir_base = 'data/data_di_network_8810.4_1000_test'
max_num_nodes =8
data_dir_undi = os.path.join(data_dir_base,'undirected/chromaticed/{}'.format(num_color))


print('data:{} \nnum_color:{}'.format(data_dir_undi,num_color))
print('\n','*'*20,'greedy:','*'*20)
# choose from: 'largest_first' 'smallest_last'. interchange = True or False
envaluate_greedy(data_dir_undi,strategy = 'smallest_last',interchange = True, num_color = int(num_color))


print('\n','*'*20,'our:','*'*20)
print('data:',data_dir_base)
base_model_save_dir = 'model_save'
envaluate_RL(data_dir_undi, num_color,max_num_nodes, device= 0, base_model_save_dir = base_model_save_dir)


print('\n','*'*20,'Tabu:','*'*20)
Tabu(data_dir_undi, num_color)