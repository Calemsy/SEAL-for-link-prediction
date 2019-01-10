from sys import path
path.append(r'./GNN_implement/')
from GNN_implement.main import parse_args, gnn
from GNN_implement.gnn import split_train_test, train
path.append(r"./node2vec/src/")
import numpy as np
import networkx as nx
from sklearn import metrics
import node2vec
from gensim.models import Word2Vec
from operator import itemgetter
from tqdm import tqdm


def load_data(data_name, network_type):
    """
    :param data_name: 
    :param network_type: use 0 and 1 stands for undirected or directed graph, respectively
    :return: 
    """
    print("load data...")
    file_path = "./raw_data/" + data_name + ".txt"
    positive = np.loadtxt(file_path, dtype=int, usecols=(0, 1))

    # sample negative
    G = nx.Graph() if network_type == 0 else nx.DiGraph()
    G.add_edges_from(positive)
    print(nx.info(G))
    negative_all = list(nx.non_edges(G))
    np.random.shuffle(negative_all)
    negative = np.asarray(negative_all[:len(positive)])
    print("positve examples: %d, negative examples: %d." % (len(positive), len(negative)))
    np.random.shuffle(positive)
    if np.min(positive) == 1:
        positive -= 1
        negative -= 1
    return positive, negative, len(G.nodes())


def learning_embedding(positive, negative, network_size, test_ratio, dimension, network_type, negative_injection=True):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param network_size: scalar, nodes size in the network
    :param test_ratio: proportion of the test set 
    :param dimension: size of the node2vec
    :param network_type: directed or undirected
    :param negative_injection: add negative edges to learn word embedding
    :return: 
    """
    print("learning embedding...")
    # used training data only
    test_size = int(test_ratio * positive.shape[0])
    train_posi, train_nega = positive[:-test_size], negative[:-test_size]
    # negative injection
    A = nx.Graph() if network_type == 0 else nx.DiGraph()
    A.add_weighted_edges_from(np.concatenate([train_posi, np.ones(shape=[train_posi.shape[0], 1], dtype=np.int8)], axis=1))
    if negative_injection:
        A.add_weighted_edges_from(np.concatenate([train_nega, np.ones(shape=[train_nega.shape[0], 1], dtype=np.int8)], axis=1))
    # node2vec
    G = node2vec.Graph(A, is_directed=False if network_type == 0 else True, p=1, q=1)
    G.preprocess_transition_probs()
    walks = G.simulate_walks(num_walks=10, walk_length=80)
    walks = [list(map(str, walk)) for walk in walks]
    model = Word2Vec(walks, size=dimension, window=10, min_count=0, sg=1, workers=8, iter=1)
    wv = model.wv
    embedding_feature, empty_indices, avg_feature = np.zeros([network_size, dimension]), [], 0
    for i in range(network_size):
        if str(i) in wv:
            embedding_feature[i] = wv.word_vec(str(i))
            avg_feature += wv.word_vec(str(i))
        else:
            empty_indices.append(i)
    embedding_feature[empty_indices] = avg_feature / (network_size - len(empty_indices))
    print("embedding feature shape: ", embedding_feature.shape)
    return embedding_feature


def link2subgraph(positive, negative, nodes_size, test_ratio, hop, network_type, max_neighbors=100):
    """
    :param positive: ndarray, from 'load_data', all positive edges
    :param negative: ndarray, from 'load_data', all negative edges
    :param nodes_size: int, scalar, nodes size in the network
    :param test_ratio: float, scalar, proportion of the test set 
    :param hop: option: 0, 1, 2, ..., or 'auto'
    :param network_type: directed or undirected
    :param max_neighbors: 
    :return: 
    """
    print("extract enclosing subgraph...")
    test_size = int(len(positive) * test_ratio)
    train_pos, test_pos = positive[:-test_size], positive[-test_size:]
    train_neg, test_neg = negative[:-test_size], negative[-test_size:]

    A = np.zeros([nodes_size, nodes_size], dtype=np.uint8)
    A[train_pos[:, 0], train_pos[:, 1]] = 1
    if network_type == 0:
        A[train_pos[:, 1], train_pos[:, 0]] = 1

    def calculate_auc(scores, test_pos, test_neg):
        pos_scores = scores[test_pos[:, 0], test_pos[:, 1]]
        neg_scores = scores[test_neg[:, 0], test_neg[:, 1]]
        s = np.concatenate([pos_scores, neg_scores])
        y = np.concatenate([np.ones(len(test_pos), dtype=np.int8), np.zeros(len(test_neg), dtype=np.int8)])
        assert len(s) == len(y)
        auc = metrics.roc_auc_score(y_true=y, y_score=s)
        return auc

    # determine the h value
    if hop == "auto":
        def cn():
            return np.matmul(A, A)
        def aa():
            A_ = A / np.log(A.sum(axis=1))
            A_[np.isnan(A_)] = 0
            A_[np.isinf(A_)] = 0
            return A.dot(A_)
        cn_scores, aa_scores = cn(), aa()
        cn_auc = calculate_auc(cn_scores, test_pos, test_neg)
        aa_auc = calculate_auc(aa_scores, test_pos, test_neg)
        if cn_auc > aa_auc:
            print("cn(first order heuristic): %f > aa(second order heuristic) %f." % (cn_auc, aa_auc))
            hop = 1
        else:
            print("aa(second order heuristic): %f > cn(first order heuristic) %f. " % (aa_auc, cn_auc))
            hop = 2

    print("hop = %s." % hop)

    # extract the subgraph for (positive, negative)
    G = nx.Graph() if network_type == 0 else nx.DiGraph()
    G.add_nodes_from(set(positive[:, 0]) | set(positive[:, 1]) | set(negative[:, 0]) | set(negative[:, 1]))
    # G.add_nodes_from(set(sum(positive.tolist(), [])) | set(sum(negative.tolist(), [])))
    G.add_edges_from(train_pos)

    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes = [], [], [], [], []
    for graph_label, data in enumerate([negative, positive]):
        print("for %s. " % "negative" if graph_label == 0 else "positive")
        for node_pair in tqdm(data):
            sub_nodes, sub_adj, vertex_tag = extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors)
            graphs_adj.append(sub_adj)
            vertex_tags.append(vertex_tag)
            node_size_list.append(len(vertex_tag))
            sub_graphs_nodes.append(sub_nodes)
    assert len(graphs_adj) == len(vertex_tags) == len(node_size_list)
    labels = np.concatenate([np.zeros(len(negative), dtype=np.uint8), np.ones(len(positive), dtype=np.uint8)]).reshape(-1, 1)

    # vertex_tags_set = list(set(sum(vertex_tags, [])))
    vertex_tags_set = set()
    for tags in vertex_tags:
        vertex_tags_set = vertex_tags_set.union(set(tags))
    vertex_tags_set = list(vertex_tags_set)
    tags_size = len(vertex_tags_set)
    print("tight the vertices tags.")
    if set(range(len(vertex_tags_set))) != set(vertex_tags_set):
        vertex_map = dict([(x, vertex_tags_set.index(x)) for x in vertex_tags_set])
        for index, graph_tag in tqdm(enumerate(vertex_tags)):
            vertex_tags[index] = list(itemgetter(*graph_tag)(vertex_map))
    return graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size


def extract_subgraph(node_pair, G, A, hop, network_type, max_neighbors):
    """
    :param node_pair:  (vertex_start, vertex_end)
    :param G:  nx object from the positive edges
    :param A:  equivalent to the G, adj matrix of G
    :param hop:
    :param network_type:
    :param max_neighbors: 
    :return: 
        sub_graph_nodes: use for select the embedding feature
        sub_graph_adj: adjacent matrix of the enclosing sub-graph
        vertex_tag: node type information from the labeling algorithm
    """
    sub_graph_nodes = set(node_pair)
    nodes = list(node_pair)

    for i in range(int(hop)):
        np.random.shuffle(nodes)
        for node in nodes:
            neighbors = list(nx.neighbors(G, node))
            if len(sub_graph_nodes) + len(neighbors) < max_neighbors:
                sub_graph_nodes = sub_graph_nodes.union(neighbors)
            else:
                np.random.shuffle(neighbors)
                sub_graph_nodes = sub_graph_nodes.union(neighbors[:max_neighbors - len(sub_graph_nodes)])
                break
        nodes = sub_graph_nodes - set(nodes)
    sub_graph_nodes.remove(node_pair[0])
    if node_pair[0] != node_pair[1]:
        sub_graph_nodes.remove(node_pair[1])
    sub_graph_nodes = [node_pair[0], node_pair[1]] + list(sub_graph_nodes)
    sub_graph_adj = A[sub_graph_nodes, :][:, sub_graph_nodes]
    sub_graph_adj[0][1] = sub_graph_adj[1][0] = 0

    # labeling(coloring/tagging)
    vertex_tag = node_labeling(sub_graph_adj, network_type)
    return sub_graph_nodes, sub_graph_adj, vertex_tag


def node_labeling(graph_adj, network_type):
    nodes_size = len(graph_adj)
    G = nx.Graph(data=graph_adj) if network_type == 0 else nx.DiGraph(data=graph_adj)
    if len(G.nodes()) == 0:
        return [1, 1]
    tags = []
    for node in range(2, nodes_size):
        try:
            dx = nx.shortest_path_length(G, 0, node)
            dy = nx.shortest_path_length(G, 1, node)
        except nx.NetworkXNoPath:
            tags.append(0)
            continue
        d = dx + dy
        div, mod = np.divmod(d, 2)
        tag = 1 + np.min([dx, dy]) + div * (div + mod - 1)
        tags.append(tag)
    return [1, 1] + tags


def create_input_for_gnn_fly(graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes,
                             embedding_feature, explicit_feature, tags_size):
    print("create input for gnn on fly, (skipping I/O operation)")
    # graphs, nodes_size_list, labels = data["graphs"], data["nodes_size_list"], data["labels"]

    # 1 - prepare Y
    Y = np.where(np.reshape(labels, [-1, 1]) == 1, 1, 0)
    print("positive examples: %d, negative examples: %d." % (np.sum(Y == 0), np.sum(Y == 1)))
    # 2 - prepare A_title
    # graphs_adj is A_title in the formular of Graph Convolution layer
    # add eye to A_title
    for index, x in enumerate(graphs_adj):
        graphs_adj[index] = x + np.eye(x.shape[0], dtype=np.uint8)
    # 3 - prepare D_inverse
    D_inverse = []
    for x in graphs_adj:
        D_inverse.append(np.linalg.inv(np.diag(np.sum(x, axis=1))))
    # 4 - prepare X
    X, initial_feature_channels = [], 0

    def convert_to_one_hot(y, C):
        return np.eye(C, dtype=np.uint8)[y.reshape(-1)]

    if vertex_tags is not None:
        initial_feature_channels = tags_size
        print("X: one-hot vertex tag, tag size %d." % initial_feature_channels)
        for tag in vertex_tags:
            x = convert_to_one_hot(np.array(tag), initial_feature_channels)
            X.append(x)
    else:
        print("X: normalized node degree.")
        for graph in graphs_adj:
            degree_total = np.sum(graph, axis=1)
            X.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))
        initial_feature_channels = 1
    X = np.array(X)
    if embedding_feature is not None:
        print("embedding feature has considered")
        # build embedding for enclosing sub-graph
        sub_graph_emb = []
        for sub_nodes in sub_graphs_nodes:
            sub_graph_emb.append(embedding_feature[sub_nodes])
        for i in range(len(X)):
            X[i] = np.concatenate([X[i], sub_graph_emb[i]], axis=1)
        initial_feature_channels = len(X[0][0])
    if explicit_feature is not None:
        initial_feature_channels = len(X[0][0])
        pass
    print("so, initial feature channels: ", initial_feature_channels)
    return np.array(D_inverse), graphs_adj, Y, X, node_size_list, initial_feature_channels  # ps, graph_adj is A_title


def classifier(data_name, is_directed, test_ratio, dimension, hop, learning_rate, top_k=60, epoch=100):
    positive, negative, nodes_size = load_data(data_name, is_directed)
    embedding_feature = learning_embedding(positive, negative, nodes_size, test_ratio, dimension, is_directed)
    graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, tags_size = \
        link2subgraph(positive, negative, nodes_size, test_ratio, hop, is_directed)

    D_inverse, A_tilde, Y, X, nodes_size_list, initial_feature_dimension = create_input_for_gnn_fly(
        graphs_adj, labels, vertex_tags, node_size_list, sub_graphs_nodes, embedding_feature, None, tags_size)
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
    nodes_size_list_train, nodes_size_list_test = split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list)

    print("data set: ", data_name)
    print("show configure for gnn.")
    print("number of enclosing sub-graph is: ", len(graphs_adj))
    print("size of vertices tags is: ", tags_size)
    print("in all enclosing sub-graph, max nodes is %d, min nodes is %d, average node is %.2d." % (
        np.max(node_size_list), np.min(node_size_list), np.average(node_size_list)))

    test_acc, prediction, pos_scores = train(X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train,
                     X_test, D_inverse_test, A_tilde_test, Y_test, nodes_size_list_test,
                     top_k, initial_feature_dimension, learning_rate, epoch)
    auc = metrics.roc_auc_score(y_true=np.squeeze(Y_test), y_score=np.squeeze(pos_scores))
    print("auc: %f" % auc)
    return auc
