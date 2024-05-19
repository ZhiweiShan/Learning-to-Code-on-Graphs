import os
from envaluate.Envaluate_local_color import envaluate_RL_local_same_model

num_color = 5
clean2 = 1
data_dir_base = 'data/data_di_bipar_20200.20.2_100_test'
max_num_nodes = 20
data_dir_di = os.path.join(data_dir_base,'directed/chromaticed/{}'.format(num_color))

print('data:{} \nnum_color:{}'.format(data_dir_di,num_color))
save_graph_picture = False

base_model_save_dir = 'model_save'
training_device = 0

cum_success_color_count, cum_success_local_color_count, cum_cnt, run_time = envaluate_RL_local_same_model(data_dir_di,
           num_color,max_num_nodes, device= 0, training_device = training_device, base_model_save_dir = base_model_save_dir,
           save_graph = save_graph_picture, clean2 = clean2)
print('time:',run_time)
