import os
import pickle
from fractions import Fraction
import numpy as np
from collections import Counter
import networkx as nx


def get_key(val,dict):
    for key, values in dict.items():
        if val in values:
            return key
    return None

def get_fraction(str_value):
    if str_value is not None:
        nums = list(map(int, str_value.split('-')))
        fraction = Fraction(nums[0], nums[1])
        decimal_value = float(fraction)
        return decimal_value
    else:
        return 0

def simplify_fraction(str_value):
    if str_value is not None:
        nums = list(map(int, str_value.split('-')))
        if nums[0] == 2 and nums[1] % 2 == 0:
            nums[1] = int(nums[1]/2)
            nums[0] = int(nums[0] / 2)
        return '{}-{}'.format(nums[0],nums[1])
    else:
        return '0'
def check_bound(dict):
    common_items_count = 0
    local_vars = locals()
    for key, value in dict.items():
        for item in value:
            if simplify_fraction(key) in MAIS_dict and item in MAIS_dict[simplify_fraction(key)]:
                common_items_count += 1
    return common_items_count


data_dir_base = '../data/new_data_di_ER_6_0.4_1000_test_chromatic' #'data/new_data_di_network_151510.2_1000_test'
print('data_dir_base',data_dir_base)

result_dir = os.path.join(data_dir_base,'MAIS.pickle')
with open(result_dir, 'rb') as f:
    MAIS_dict = pickle.load(f)

result_dir = os.path.join(data_dir_base,'OSIA.pickle')
with open(result_dir, 'rb') as f:
    OSIA_dict = pickle.load(f)

result_dir = os.path.join(data_dir_base, 'orthogonal.pickle')
with open(result_dir, 'rb') as f:
    orthogonal_dict = pickle.load(f)

result_dir = os.path.join(data_dir_base,'OVIA_b2.pickle')
with open(result_dir, 'rb') as f:
    OVIA_b2_dict = pickle.load(f)

# result_dir = os.path.join(data_dir_base,'OVIA_b3.pickle')
# with open(result_dir, 'rb') as f:
#     OVIA_b3_dict = pickle.load(f)

result_dir = os.path.join(data_dir_base,'SSIA.pickle')
with open(result_dir, 'rb') as f:
    SSIA_dict = pickle.load(f)
try:
    result_dir = os.path.join(data_dir_base,'SVIA_b2.pickle')
    with open(result_dir, 'rb') as f:
        SVIA_b2_dict = pickle.load(f)
except:
    pass

try:
    result_dir = os.path.join(data_dir_base,'SIMO_scalar.pickle')
    with open(result_dir, 'rb') as f:
        SIMO_scalar_dict = pickle.load(f)
except:
    pass

# count the number of graphs reaching upper bound from MAIS
common_items_count = check_bound(orthogonal_dict)
print('The number of graphs reaching upper bound from MAIS by {}: {}'.format('othgn',common_items_count))
common_items_count = check_bound(OSIA_dict)
print('The number of graphs reaching upper bound from MAIS by {}: {}'.format('OSIA',common_items_count))
common_items_count = check_bound(OVIA_b2_dict)
print('The number of graphs reaching upper bound from MAIS by {}: {}'.format('OVIA_b2',common_items_count))
common_items_count = check_bound(SSIA_dict)
print('The number of graphs reaching upper bound from MAIS by {}: {}'.format('SSIA',common_items_count))
try:
    common_items_count = check_bound(SVIA_b2_dict)
    print('The number of graphs reaching upper bound from MAIS by {}: {}'.format('SVIA_b2',common_items_count))
except:
    pass



best_dof = np.array([])
MAIS_dof = np.array([])
orthogonal_dof = np.array([])
OSIA_dof = np.array([])
OVIA_dof = np.array([])
SSIA_dof = np.array([])
best_method = np.array([])
reach_bound = []
# 1:orthogonal 2:OSIA 3:OVIA 4:SSIA 5:SVIA
for idx in range(1000):
    res_from_SVIA = 0
    key_MAIS = get_fraction(get_key(idx, MAIS_dict))
    res_from_orthogonal = get_fraction(get_key(idx, orthogonal_dict))
    res_from_OSIA = get_fraction(get_key(idx, OSIA_dict))
    res_from_OVIA = get_fraction(get_key(idx, OVIA_b2_dict))
    #res_from_OVIA_b3 = get_fraction(get_key(idx, OVIA_b3_dict))
    #res_from_OVIA = np.max((res_from_OVIA_b2,res_from_OVIA_b3))
    res_from_SSIA = get_fraction(get_key(idx, SSIA_dict))
    #res_from_SVIA = get_fraction(get_key(idx, SVIA_b2_dict))
    res_best = np.max((res_from_orthogonal,res_from_OSIA,res_from_OVIA,res_from_SSIA))

    best_dof = np.append(best_dof,res_best)
    orthogonal_dof = np.append(orthogonal_dof, res_from_orthogonal)
    OSIA_dof = np.append(OSIA_dof, res_from_OSIA)
    OVIA_dof = np.append(OVIA_dof, res_from_OVIA)
    SSIA_dof = np.append(SSIA_dof, res_from_SSIA)
    MAIS_dof = np.append(MAIS_dof, key_MAIS)
    # check how many cases reaching upper bound

    if res_best == key_MAIS:
        reach_bound.append(idx)

    if res_from_orthogonal == res_best:
        best_method = np.append(best_method, 1)
        continue
    elif res_from_OSIA == res_best:
        best_method = np.append(best_method, 2)
        continue
    elif res_from_OVIA == res_best:
        best_method = np.append(best_method, 3)
        continue
    elif res_from_SSIA == res_best:
        best_method = np.append(best_method, 4)
        continue
    elif res_from_SVIA == res_best:
        best_method = np.append(best_method, 5)
        continue
    else:
        print('error')

#print('OVIA_b2_or_b3>OSIA',np.count_nonzero(best_method>1))
best_method_list = best_method.tolist()
counts = Counter(best_method_list)
print('best_method count:', counts)
print('num of cases reaching MAIS', len(reach_bound))

# for SIMO:
import matplotlib.pyplot as plt
dof_SIMO = np.array([])
try:
    for idx in range(1000):
        res_from_SIMO_scalar = get_fraction(get_key(idx, SIMO_scalar_dict))
        dof_SIMO = np.append(dof_SIMO, res_from_SIMO_scalar)
except:
    pass



# Plot
num_grapg_plot = 50
instances = np.arange(1, num_grapg_plot+1)
plt.figure(figsize=(10, 6))


plt.scatter(instances, best_dof[0:num_grapg_plot], label='SISO', color='#2EC4B6', marker='o')
plt.scatter(instances, dof_SIMO[0:num_grapg_plot], label='SIMO-(1,2)', color='#E71D36', marker='x')

plt.xlabel('Instance No.')
plt.ylabel('Achieved DoF')
plt.title('Comparison of Method A and Method B Results')
plt.legend()
plt.grid(True)
plt.show()


print('num of SIMO>SISO:',np.shape(np.where(dof_SIMO[0:num_grapg_plot]>best_dof[0:num_grapg_plot]))[1])

all_numbers = set(range(1000))
all_numbers - set(reach_bound)
for index in all_numbers - set(reach_bound):
    index = 461
    read_dir = os.path.join(data_dir_base,'directed', "{:06d}.txt".format(index))
    plot_save_path = os.path.join(data_dir_base, "{}_gap_bound".format(index))
    g = nx.relabel.convert_node_labels_to_integers(
        nx.readwrite.edgelist.read_edgelist(read_dir, create_using=nx.DiGraph),
        first_label=0)
    nx.draw_networkx(g, pos=nx.circular_layout(g), with_labels=True, font_weight='bold')
    plt.savefig(plot_save_path)
    plt.show()
    print('index:',index)
    print('MAIS_bound', MAIS_dof[index])
    print('orthogonal_dof', orthogonal_dof[index])
    print('OSIA_dof', OSIA_dof[index])
    #print('OVIA_dof', OVIA_dof[index])
    print('SSIA_dof', SSIA_dof[index])
    print('best_dof_SIMO', dof_SIMO[index])

np.where(dof_SIMO==best_dof)
np.shape(np.where(dof_SIMO/best_dof == 2))


