# Learning-to-Code-on-Graphs
Official codes for the paper "Learning to Code on Graphs for Topological Interference Management".

Zhiwei Shan, Xinping Yi, Han Yu, Chung-Shou Liao, and Shi Jin
## Abstract
The state-of-the-art coding schemes for topological interference management (TIM) problems are usually handcrafted for specific families of network topologies, relying critically on experts’ domain knowledge. This inevitably restricts the potential wider applications to wireless communication systems, due to the limited generalizability. This work makes the first attempt to advocate a novel intelligent coding approach to mimic topological interference alignment (IA) via local graph coloring algorithms, leveraging the new advances of graph neural networks (GNNs) and reinforcement learning (RL). The proposed LCG framework is then generalized to discover new IA coding schemes, including one-to-one vector IA and subspace IA. The extensive experiments demonstrate the excellent generalizability and transferability of the proposed approach, where the parameterized GNNs trained by small size TIM instances are able to work well on new unseen network topologies with larger size.

## Requirements

- Python 3.7
- CUDA 11.3
- PyTorch 1.12.0
- DGL 0.9.1
- MATLAB R2020b
- MATLAB Engine for Python
- Matplotlib
- Exactor

## Setup
```python
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch
conda install -c dglteam "dgl-cuda11.3=0.9.1"
conda install matplotlib
```
Install the MATLAB Engine for Python. Refer to the official MATLAB documentation for installation instructions. It is used to generate the wireless network simulation graphs. One may skip MATLAB if wireless network graphs are not needed.

Install the Exactor following the link: https://github.com/ZhiweiShan/exactcolors. It is used to calculate the chromatic numbers of graphs.

Set the Exactor path you installed in the file 'generate_data/Run_exact_di_timelimit', line 55.

## Usage
1. Create Training Data and Testing Data
To create training data and testing data, run the following command:
```
python create_data_di.py
```
By default, it generates random ER bipartite graphs. You can set your own parameters to generate more datasets.

2. Train
To train the model, run the following command:
```
python train_color.py
```
Several parameters need to be seted. An example is:
```
python train_color.py --data-dir=data/data_di_bipar_20200.20.2_100/undirected/chromaticed/5 --model-save-dir=model_save --device=0 --num-color=5 --num-nodes=20
```
Note that ‘num-nodes‘ only needs to be greater or equal to the number of nodes of the graph with the maximum number of nodes in the dataset.

3. Test Coloring
To test the graph coloring algorithm, run the following command:
```
python Run_Envaluate_color_compair.py
```

4. Test Local Coloring
To test the local coloring algorithm, run the following command:
```
python Run_Envaluate_local_color.py
```

5. Test Fractional Local Coloring
First, generate splitting graphs:
```
python generate_data/Node_splitting.py
```
Then, run the following command to evaluate the fractional local coloring algorithm:
```
python Run_Envaluate_fractional_local_color.py
```

6. Test Matrix Rank Reduction
```
python Run_Envaluate_matrix_rank_reduction.py
```
