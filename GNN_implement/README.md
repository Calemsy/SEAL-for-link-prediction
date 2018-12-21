# Deep Graph Convolutional Neural Network implement by tensorflow

## 1 - About

GNN is a novel and powerful deep neural network for graph classification, It usually consists of *(1)*`graph convolution layer` which extract local substructure features for individual links and *(2)* a `SortPooling layer` which aggregates node-level features into a graph-level feature vector. It's directly accepts graph data as input without the need of first transforming graphs into tensors, make end-to-end gradient-based training possible. And it enables learning from global topology by sorting the vertex features instead of summing them up, which is supportd by `SortPooling layer`.

This repository provides a reference implementation of GNN based on **Tensorflow**.

For more information, please refer to:
> M. Zhang, Z. Cui, M. Neumann, and Y. Chen, An End-to-End Deep Learning Architecture for Graph Classification, Proc. AAAI Conference on Artificial Intelligence (AAAI-18).
and the origal PyTorch implementation of DGCNN is [here](https://github.com/muhanzhang/pytorch_DGCNN)

## 2 - Basic Usage

### 2.1 - Example

To run the `GNN` on the `mutag`(default setting), type the following command on the home directory:

`python main.py`

### 2.2 - Options

 - `python main.py --data proteins` to run `GNN` on proteins
 - `python main.py --epoch 200` to assign the number of epochs, default value is 100
 - `python main.py -r 0.00001` or `python main.py --learning_rate 0.00001` to set the learning rate which determine the speed of update.
 - ...
 
you can check out the other options available to use `python main.py --help`

## 3 - Result

| **Dataset**  | Mutag  | NCI1 | PROTEINS | D&D |
|:------:|:------:|:------:|:------:|:------:|
|Nodes(max) |   28    |  111    | 620   | 5748   |
|Nodes(avg.)|  17.93  |  29.87  | 39.06 | 284.32 |
|Nodes(min) |   10    |    3    | 4     |   30   |
|Graphs     |   188   |   4110  | 1113  |   1178 |
|**GNN**|**0.8684**(0.058844)|**0.7073**(0.018595)|**0.7509**(0.027505)|**0.7432**(0.047040)|

To be continued...
