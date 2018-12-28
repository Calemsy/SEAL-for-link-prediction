import argparse
from seal_link_predict import *


def parse_args(data="USAir", epoch=100, lr=0.00001, is_directed=0):
    parser = argparse.ArgumentParser(description="Link prediction with SEAL.")
    parser.add_argument("--data", type=str, help="data name.", default=data)
    parser.add_argument("--epoch", type=int, default=epoch, help="epochs of gnn")
    parser.add_argument("--learning_rate", type=float, default=lr, help="learning rate")
    parser.add_argument("--is_directed",  type=int, default=is_directed, help="use 0, 1 stands for undirected or directed graph")
    parser.add_argument("--test_ratio", type=float, default=0.1, help="ratio of testing set")
    parser.add_argument("--hop", default="auto", help="option: 0, 1, ... or 'auto'.")
    parser.add_argument("--dimension", default=128, type=int, help="number of embedding.")
    return parser.parse_args()


def seal(args):
    print("data set: ", args.data)
    positive, negative, nodes_size = load_data(args.data, args.is_directed)
    embedding_feature = \
        learning_embedding(positive, negative, nodes_size, args.test_ratio, args.dimension, args.is_directed)
    graphs_adj, labels, vertex_tags, node_size_list, sub_graph_nodes = \
        link2subgraph(positive, negative, nodes_size, args.test_ratio, args.hop, args.is_directed)
    create_input_for_gnn(graphs_adj, labels, vertex_tags, node_size_list,
                         sub_graph_nodes, embedding_feature, None, args.data)
    return classifier(args.data, args.epoch, args.learning_rate, args.is_directed)


def link_predict():
    args = parse_args()
    seal(args)

if __name__ == "__main__":
    link_predict()