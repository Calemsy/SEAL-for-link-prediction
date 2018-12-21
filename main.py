import argparse
from seal_link_predict import *


parser = argparse.ArgumentParser(description="Link prediction with SEAL.")
parser.add_argument("--data", type=str, help="data name.", default="USAir")
parser.add_argument("--epoch", type=int, help="epochs of gnn", default=100)
parser.add_argument("--network_type",  type=int, default=0, help="use 0, 1 stands for undirected or directed graph")
parser.add_argument("--test_ratio", type=float, default=0.1, help="ratio of testing set")
parser.add_argument("--hop", default="auto", help="option: 0, 1, ... or 'auto'.")
parser.add_argument("--dimension", default=128, type=int, help="number of embedding.")
args = parser.parse_args()


def seal():
    print("data set: ", args.data)
    positive, negative, nodes_size = load_data(args.data, args.network_type)
    embedding_feature = \
        learning_embedding(positive, negative, nodes_size, args.test_ratio, args.dimension, args.network_type)
    graphs_adj, labels, vertex_tags, node_size_list, sub_graph_nodes = \
        link2subgraph(positive, negative, nodes_size, args.test_ratio, args.hop, args.network_type)
    create_input_for_gnn(graphs_adj, labels, vertex_tags, node_size_list,
                         sub_graph_nodes, embedding_feature, None, args.data)
    classifier(args.data)


if __name__ == "__main__":
    seal()