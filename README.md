# Learning-to-Code-on-Graphs
Official codes for the paper "Learning to Code on Graphs for Topological Interference Management".

Zhiwei Shan, Xinping Yi, Han Yu, Chung-Shou Liao, and Shi Jin

Updated for our journal version paper (under review)
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
### For quik test
One can use the generated small data sets and skip the installation of MATLAB and Exactor, jump to Usage 2. 
### Build data
If you want to test wireless network graphs:
Install the MATLAB Engine for Python. Refer to the official MATLAB documentation for installation instructions. It is used to generate the wireless network simulation graphs.


Install the Exactor following the link: https://github.com/ZhiweiShan/exactcolors. It is used to calculate the chromatic numbers of graphs.

Set the Exactor path you installed in the file 'generate_data/Run_exact', line 56.

## Usage
1. Create Training Data and Testing Data
To create training data and testing data, run the following command:
```
python create_data_di.py
```
You can set your own parameters to generate more datasets.

2. Train

To train the model, run the following command:
```
python train_color.py
```
Several parameters need to be seted. An example is:
```
python train_color.py --data-dir=data/data_di_bipar_20200.20.2_1000_train/undirected/chromaticed/5 --model-save-dir=model_save --device=0 --num-color=5 --num-nodes=20
```
Note that ‘num-nodes‘ only needs to be greater or equal to the number of nodes of the graph with the maximum number of nodes in the dataset.

3. Test Coloring

To test the graph coloring algorithm, run the following command:
```
python Compair.py
```

4. Test Local Coloring

To test the local coloring algorithm, run the following command:
```
python Envaluate/Run_Envaluate_local_color.py
```

5. Test Fractional Local Coloring

First, generate splitting graphs:
```
python generate_data/Node_splitting.py
```
Then, run the following command to evaluate the fractional local coloring algorithm:
```
python Envaluate/Run_Envaluate_fractional_local_color.py
```

6. Test Matrix Rank Reduction

```
python Envaluate/Run_Envaluate_matrix_rank_reduction.py
```

# Update for journal version
run the following command:
```
python Journal_compaire.py
```
to generate multiple IA coding scheme, together with TDMA and MAIS bound.

use
```
Journal/Journal_load_result
```
to compaire the achieved DoF obtained by different method

# Dataset Details
We train on graphs randomly generated based on variant specific parameters. 50,000 graphs were generated and separated according to the chromatic number for training, and 5,000 graphs for evaluation. The specific parameters for random graphs are shown in tables below. q represents the percent of randomly choosing demanded messages.
    
For Wireless Net, we randomly distribute transmitters and receivers within a square area of 1,000 m * 1,000 m.  The simulated channel follows the LoS model in ITU-1411. The carrier frequency is 2.4 GHz, antenna height is 1.5 m, and the antenna gain per device is -2.5 dB. The noise power spectral density is -174 dBm/Hz, and the noise figure is 7 dB. Each pair of transmitter and receiver is uniformly spaced within [2, 65] meters. Each link is expected to operate over a 10 MHz spectrum and the maximum transmit power is 30 dBm. 

![2024-05-20_23-45](https://github.com/ZhiweiShan/Learning-to-Code-on-Graphs/assets/74201033/9d7a4bf0-22cf-4727-83c2-e6e5a5992177)

![2024-05-20_23-46](https://github.com/ZhiweiShan/Learning-to-Code-on-Graphs/assets/74201033/69202331-e278-45ab-94b2-b6305998b35a)

