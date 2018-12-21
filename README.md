# SEAL-for-link-prediction

`SEAL, a link prediction framework based on GNN.`

## 1 - About

This repository is a reference implementation of SEAL proposed in the paper: 

>M. Zhang and Y. Chen, Link Prediction Based on Graph Neural Networks, 
Advances in Neural Information Processing Systems (NIPS-18). [Preprint](https://arxiv.org/pdf/1802.09691.pdf)

SEAL, a novel link prediction framework, to simultaneously learn from local enclosing subgraphs, embedding and attributes. 
Experimentally showed the SEAL achieved unprecedentedly strong performance by comparing to various heuristics, latent feature methods, 
and network embedding algorithms.

## 2 - Version

 - python 3.5.5</br>
 - **networkx 2.0**</br>
 - tensorflow 1.7.0</br>
 - numpy 1.14.2</br>

## 3 - Baisc Usage

#### 3.1 - Example

Type the following command to run `seal` on data `USAir`.

`python main.py`

#### 3.2 - Option

 - `python main.py --data Celegans` to run `SEAL` on data `Celegans`
 - `python main.py --epoch 200` will assign the number of epochs to 200, default value is 100
 - `python main.py -r 0.00001` will set the learning rate which determine the speed of update parameters to 0.00001.
 
you can check out the other options available to use `python main.py --help`

## 4 - Result

|Data| USAir | Celegans | Power |
|:-----:|:-----:|:-----:|:-----:|
|#Node         |332    |297    |4941|
|#Edges        |2126   |2148   |6594|
|Average Degree|12.8072|14.4646|2.6691|
|SEAL(**auc**)     |**0.9538**|**0.8979**|**0.8889**|
