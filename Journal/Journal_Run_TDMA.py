import os
import re
from shutil import copyfile
from tqdm import tqdm
import subprocess
import signal
import pickle
import sys
import helper_functions


def run_cmd(cmd_string, timeout=20):
    p = subprocess.Popen(cmd_string, stderr=subprocess.STDOUT, stdout=subprocess.PIPE, shell=True,
                         close_fds=True,
                         start_new_session=True)
    format = 'utf-8'
    try:
        (msg, errs) = p.communicate(timeout=timeout)
        ret_code = p.poll()
        if ret_code:
            code = 1
            msg = "[Error]Called Error ： " + str(msg.decode(format))
        else:
            code = 0
            msg = str(msg.decode(format))
    except subprocess.TimeoutExpired:
        p.kill()
        p.terminate()
        os.killpg(p.pid, signal.SIGUSR1)

        code = 1
        msg = "[ERROR]Timeout Error : Command '" + cmd_string + "' timed out after " + str(timeout) + " seconds"
    except Exception as e:
        code = 1
        msg = "[ERROR]Unknown Error : " + str(e)

    return code, msg


def run_exact(graph_folder_main):

    graph_folder_undi = os.path.join(graph_folder_main,'undirected')
    graph_folder_di = os.path.join(graph_folder_main, 'directed')
    graph_paths_undi = os.listdir(graph_folder_undi)
    graph_paths_di = os.listdir(graph_folder_di)
    graph_chromaticed_path_undi = os.path.join(graph_folder_undi,'chromaticed')
    os.makedirs(graph_chromaticed_path_undi,exist_ok=True)
    graph_chromaticed_path_di = os.path.join(graph_folder_di, 'chromaticed')
    os.makedirs(graph_chromaticed_path_di, exist_ok=True)

    C_list = []
    n = {}
    n_limit = 0
    for graph_path_undi in tqdm(graph_paths_undi):
        if graph_path_undi.endswith(".txt"):
            graph_path_full_undi = os.path.join(graph_folder_undi,graph_path_undi)
            # set the Exactor path bellow
            main = "exactcolors-master/color {}".format(graph_path_full_undi)

            code, msg = run_cmd(main,30)

            if code == 0:
                str_match_opt = list(filter(lambda x: "Opt Colors" in x, msg.split('\n')))
                C = int(re.findall(r"\d+", str_match_opt[0])[0])
                save_C_undi = os.path.join(graph_chromaticed_path_undi, str(C))
                os.makedirs(save_C_undi, exist_ok=True)
                save_C_di = os.path.join(graph_chromaticed_path_di, str(C))
                os.makedirs(save_C_di, exist_ok=True)
                try:
                    n[C] += 1
                except:
                    n[C] = 0
                newname_undi = save_C_undi + os.sep + "{:06d}.txt".format(n[C])
                copyfile(graph_path_full_undi, newname_undi)

                graph_path_full_di = os.path.join(graph_folder_di, graph_path_undi)
                newname_di = save_C_di + os.sep + "{:06d}.txt".format(n[C])
                copyfile(graph_path_full_di, newname_di)

                if C in C_list:
                    pass
                else:
                    C_list.append(C)
            else:
                n_limit += 1
                print('out of time limit:',n_limit)

    for C in C_list:
        graph_chromaticed_path_C_undi = os.path.join(graph_chromaticed_path_undi, str(C))
    # delet "e" and "p"
        filelist = os.listdir(graph_chromaticed_path_C_undi)
        for graph in filelist:
            file = open(os.path.join(graph_chromaticed_path_C_undi,graph),"r")
            lines = []
            for i in file:
                lines.append(i)
            file.close()

            new = []
            for line in lines:
                bit = line[0]
                if bit == "c" or "p":
                    pass
                if bit == "e":
                    new.append(line[2:])

            file_write = open(os.path.join(graph_chromaticed_path_C_undi,graph),"w")
            for var in new:
                file_write.writelines(var)
            file_write.close()

    for C in C_list:
        graph_chromaticed_path_C_di = os.path.join(graph_chromaticed_path_di, str(C))
    # delet "e" and "p"
        filelist = os.listdir(graph_chromaticed_path_C_di)
        for graph in filelist:
            file = open(os.path.join(graph_chromaticed_path_C_di,graph),"r")
            lines = []
            for i in file:
                lines.append(i)
            file.close()

            new = []
            for line in lines:
                bit = line[0]
                if bit == "c" or "p":
                    pass
                if bit == "e":
                    new.append(line[2:])

            file_write = open(os.path.join(graph_chromaticed_path_C_di,graph),"w")
            for var in new:
                file_write.writelines(var)
            file_write.close()

if __name__ == '__main__':
    data_dir_base = 'data/new_data_di_ER_6_0.3_1000_test_chromatic'
    try:
        data_dir_base = sys.argv[1]
        #data_dir_base = os.path.join('../',data_dir_base)
    except:
        pass
    data_dir_undi = os.path.join(data_dir_base, 'undirected')
    num_eval_graphs = len([
        name
        for name in os.listdir(data_dir_undi)
        if name.endswith('.txt')
    ])

    result_dir = os.path.join(data_dir_base,'orthogonal.pickle')

    achieve_dof_index = {}
    for idx in range(num_eval_graphs):
        g_path = os.path.join(data_dir_undi,
                              "{:06d}.txt".format(idx)
                              )
        # set the Exactor path bellow
        main = "exactcolors-master/color {}".format(g_path)

        code, msg = run_cmd(main,30)

        if code == 0:
            str_match_opt = list(filter(lambda x: "Opt Colors" in x, msg.split('\n')))
            C = int(re.findall(r"\d+", str_match_opt[0])[0])
            key = '1-{}'.format(C)
            if key not in achieve_dof_index:
                achieve_dof_index[key] = []
            achieve_dof_index[key].append(idx)
        else:
            print('error when running exactor')
    print('OTHGN')
    helper_functions.sort_dict(achieve_dof_index)
    with open(result_dir, 'wb') as f:
        pickle.dump(achieve_dof_index, f)