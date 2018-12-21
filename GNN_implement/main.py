import argparse
import pickle
from gnn import create_input, split_train_test, train
import numpy as np


def parse_args(data_name="mutag", epoch=100):
    parser = argparse.ArgumentParser(description="GNN(graph neural network)-tensorflow")
    parser.add_argument("-d", "--data", type=str, help="name of data", default=data_name)
    parser.add_argument("-e", "--epoch", type=int, default=epoch, help="pass through all training set call a EPOCH")
    parser.add_argument("-r", "--learning_rate", type=float, default=0.00001, help="learning rate")
    parser.add_argument("-k", "--top_k", type=int, default=60, help="for sort pooling layer to cut nodes")
    args = parser.parse_args()
    return args


def gnn(args):
    with open("./data/" + args.data + ".txt", "rb") as f_in:
        data = pickle.load(f_in)
    print("data information:")
    print("data name: ", args.data)
    print("graphs: ", len(data["graphs"]))
    print("max nodes: %d \nmin nodes: %d \naverage node %.2f" %
          (np.max(data["nodes_size_list"]), np.min(data["nodes_size_list"]), np.average(data["nodes_size_list"])))
    print("vertex tag: ", len(set(sum(data["vertex_tag"], []))))

    learning_rate = {"mutag": 0.00001, "cni1": 0.00001, "proteins": 0.000001, "dd": 0.000001}
    D_inverse, A_tilde, Y, X, nodes_size_list, initial_feature_dimension = create_input(data)
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
    nodes_size_list_train, nodes_size_list_test = split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list)

    test_acc, prediction, pos_scores = train(X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train,
                     X_test, D_inverse_test, A_tilde_test, Y_test, nodes_size_list_test,
                     args.top_k, initial_feature_dimension,
                     learning_rate[args.data] if args.learning_rate is None else args.learning_rate,
                     args.epoch, args.data)
    return test_acc, np.squeeze(prediction), np.squeeze(pos_scores), np.squeeze(Y_test)


def main():
    args = parse_args()
    gnn(args)

if __name__ == "__main__":
    main()
