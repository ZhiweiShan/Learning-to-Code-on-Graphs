from envaluate.Envaluate_local_color_split import envaluate_RL_local_same_model

orig_num_color = 5
orig_max_num_nodes = 20
num_split =2
num_color = orig_num_color* num_split
max_num_nodes = orig_max_num_nodes*num_split
data_dir_di = 'data/data_di_bipar_20200.20.2_100_test/directed/chromaticed/5_split_2'
base_model_save_dir = 'model_save/{}'.format(num_color)
training_device = 0
num_clean2 = 4
save_graph_picture = False

print('data:',data_dir_di)
print('model:',base_model_save_dir)

cum_success_local_color_count_list = []
rate = []
pre_cum_success_local = 0
index = None
for clean2 in range(1, num_clean2):
    print('\n','clean2:',clean2)

    cum_success_color_count, cum_success_local_color_count, cum_cnt, run_time, index = envaluate_RL_local_same_model(data_dir_di,
               num_color,max_num_nodes, device= 0, training_device = training_device, base_model_save_dir = base_model_save_dir,
               save_graph = save_graph_picture, clean2 = clean2, index= index)

    print('time:',run_time)


    rate.append(num_split/(num_color-clean2))
    cum_success_local_color_count_list.append(cum_success_local_color_count)
    pre_cum_success_local = cum_success_local_color_count

    print('\n success graph:', index)
    if len(index) == 0:
        break
rate = [float('{:.2f}'.format(i)) for i in rate]

print('rate: {}'.format(rate))
print('success: {}'.format(cum_success_local_color_count_list))

