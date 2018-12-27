import argparse
from seal_link_predict import *


parser = argparse.ArgumentParser(description="Link prediction with SEAL.")
parser.add_argument("--data", type=str, help="data name.", default="USAir")
parser.add_argument("--epoch", type=int, default=100, help="epochs of gnn")
parser.add_argument("--learning_rate", type=float, default=0.00001, help="learning rate")
parser.add_argument("--is_directed",  type=int, default=0, help="use 0, 1 stands for undirected or directed graph")
parser.add_argument("--test_ratio", type=float, default=0.1, help="ratio of testing set")
parser.add_argument("--hop", default="auto", help="option: 0, 1, ... or 'auto'.")
parser.add_argument("--dimension", default=128, type=int, help="number of embedding.")
cmd_args = parser.parse_args()


def seal():
    print("data set: ", cmd_args.data)
    positive, negative, nodes_size = load_data(cmd_args.data, cmd_args.is_directed)
    embedding_feature = \
        learning_embedding(positive, negative, nodes_size, cmd_args.test_ratio, cmd_args.dimension, cmd_args.is_directed)
    graphs_adj, labels, vertex_tags, node_size_list, sub_graph_nodes = \
        link2subgraph(positive, negative, nodes_size, cmd_args.test_ratio, cmd_args.hop, cmd_args.is_directed)
    create_input_for_gnn(graphs_adj, labels, vertex_tags, node_size_list,
                         sub_graph_nodes, embedding_feature, None, cmd_args.data)
    classifier(cmd_args.data, cmd_args.epoch, cmd_args.learning_rate, cmd_args.is_directed)


if __name__ == "__main__":
    seal()