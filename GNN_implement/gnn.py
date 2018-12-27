import time
import tensorflow as tf
from load_raw_data import *
from tqdm import tqdm


GRAPH_CONV_LAYER_CHANNEL = 32
CONV1D_1_OUTPUT = 16
CONV1D_2_OUTPUT = 32
CONV1D_1_FILTER_WIDTH = GRAPH_CONV_LAYER_CHANNEL * 3
CONV1D_2_FILTER_WIDTH = 5
DENSE_NODES = 128
DROP_OUTPUT_RATE = 0.5
LEARNING_RATE_BASE = 0.00004
LEARNING_RATE_DECAY = 0.99
MODEL_SAVE_PATH = "./model"
MODEL_SAVE_NAME = "gnn_model"


def create_input(data, directed):
    print("create input...")
    offset = 1 if data["index_from"] == 1 else 0
    graphs, nodes_size_list, labels = data["graphs"], data["nodes_size_list"], data["labels"]

    A_tilde, count = [], 0
    for index, graph in enumerate(graphs):
        A_tilde.append(np.zeros([nodes_size_list[index], nodes_size_list[index]], dtype=np.float32))
        for edge in graph:
            A_tilde[count][edge[0] - offset][edge[1] - offset] = 1.
            if directed == 0:
                A_tilde[count][edge[1] - offset][edge[0] - offset] = 1.
        count += 1
    Y = np.where(np.reshape(labels, [-1, 1]) == 1, 1, 0)
    print("positive examples: %d, negative examples: %d." % (np.sum(Y == 0), np.sum(Y == 1)))
    A_tilde = np.array(A_tilde)

    # get A_title
    for index, x in enumerate(A_tilde):
        A_tilde[index] = x + np.eye(x.shape[0])
    # get D_inverse
    D_inverse = []
    for x in A_tilde:
        D_inverse.append(np.linalg.inv(np.diag(np.sum(x, axis=1))))
    # get X
    X, initial_feature_channels = [], 0
    def convert_to_one_hot(y, C):
        return np.eye(C)[y.reshape(-1)]
    if data["vertex_tag"]:
        vertex_tag = data["vertex_tag"]
        initial_feature_channels = len(set(sum(vertex_tag, [])))
        print("X: one-hot vertex tag, tag size %d." % (initial_feature_channels))
        for tag in vertex_tag:
            x = convert_to_one_hot(np.array(tag) - offset, initial_feature_channels)
            X.append(x)
    else:
        print("X: normalized node degree.")
        for graph in A_tilde:
            degree_total = np.sum(graph, axis=1)
            X.append(np.divide(degree_total, np.sum(degree_total)).reshape(-1, 1))
        initial_feature_channels = 1
    X = np.array(X)
    if data["feature"] is not None:
        print("embedding or explicit feature.")
        feature = data["feature"]
        for i in range(len(X)):
            X[i] = np.concatenate([X[i], feature[i]], axis=1)
        initial_feature_channels = len(X[0][0])
    print("initial feature channels: ", initial_feature_channels)
    return np.array(D_inverse), A_tilde, Y, X, nodes_size_list, initial_feature_channels


def split_train_test(D_inverse, A_tilde, X, Y, nodes_size_list, rate=0.1):
    print("split training and test data...")
    state = np.random.get_state()
    np.random.shuffle(D_inverse)
    np.random.set_state(state)
    np.random.shuffle(A_tilde)
    np.random.set_state(state)
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    np.random.set_state(state)
    np.random.shuffle(nodes_size_list)
    data_size = Y.shape[0]
    training_set_size, test_set_size = int(data_size * (1 - rate)), int(data_size * rate)
    D_inverse_train, D_inverse_test = D_inverse[: training_set_size], D_inverse[training_set_size:]
    A_tilde_train, A_tilde_test = A_tilde[: training_set_size], A_tilde[training_set_size:]
    X_train, X_test = X[: training_set_size], X[training_set_size:]
    Y_train, Y_test = Y[: training_set_size], Y[training_set_size:]
    nodes_size_list_train, nodes_size_list_test = nodes_size_list[: training_set_size], nodes_size_list[training_set_size:]
    print("about train: positive examples(%d): %s, negative examples: %s."
          % (training_set_size, np.sum(Y_train == 1), np.sum(Y_train == 0)))
    print("about test: positive examples(%d): %s, negative examples: %s."
          % (test_set_size, np.sum(Y_test == 1), np.sum(Y_test == 0)))
    return D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test, \
           nodes_size_list_train, nodes_size_list_test


def train(X_train, D_inverse_train, A_tilde_train, Y_train, nodes_size_list_train,
        X_test, D_inverse_test, A_tilde_test, Y_test, nodes_size_list_test,
        top_k, initial_channels, learning_rate=0.00001, epoch=100, data_name="mutag", debug=False):

    # placeholder
    D_inverse_pl = tf.placeholder(dtype=tf.float32, shape=[None, None])
    A_tilde_pl = tf.placeholder(dtype=tf.float32, shape=[None, None])
    X_pl = tf.placeholder(dtype=tf.float32, shape=[None, initial_channels])
    Y_pl = tf.placeholder(dtype=tf.int32, shape=[1], name="Y-placeholder")
    node_size_pl = tf.placeholder(dtype=tf.int32, shape=[], name="node-size-placeholder")
    is_train = tf.placeholder(dtype=tf.uint8, shape=[], name="is-train-or-test")

    # trainable parameters of graph convolution layer
    graph_weight_1 = tf.Variable(tf.truncated_normal(shape=[initial_channels, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32))
    graph_weight_2 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32))
    graph_weight_3 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, GRAPH_CONV_LAYER_CHANNEL], stddev=0.1, dtype=tf.float32))
    graph_weight_4 = tf.Variable(tf.truncated_normal(shape=[GRAPH_CONV_LAYER_CHANNEL, 1], stddev=0.1, dtype=tf.float32))

    # GRAPH CONVOLUTION LAYER
    gl_1_XxW = tf.matmul(X_pl, graph_weight_1)
    gl_1_AxXxW = tf.matmul(A_tilde_pl, gl_1_XxW)
    Z_1 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_1_AxXxW))
    gl_2_XxW = tf.matmul(Z_1, graph_weight_2)
    gl_2_AxXxW = tf.matmul(A_tilde_pl, gl_2_XxW)
    Z_2 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_2_AxXxW))
    gl_3_XxW = tf.matmul(Z_2, graph_weight_3)
    gl_3_AxXxW = tf.matmul(A_tilde_pl, gl_3_XxW)
    Z_3 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_3_AxXxW))
    gl_4_XxW = tf.matmul(Z_3, graph_weight_4)
    gl_4_AxXxW = tf.matmul(A_tilde_pl, gl_4_XxW)
    Z_4 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_4_AxXxW))
    graph_conv_output = tf.concat([Z_1, Z_2, Z_3], axis=1)  # shape=(node_size/None, 32+32+32)

    def variable_summary(var):
        var_mean = tf.reduce_mean(var)
        var_variance = tf.square(tf.reduce_mean(tf.square(var - tf.reduce_mean(var))))
        var_max = tf.reduce_max(var)
        var_min = tf.reduce_min(var)
        return var_mean, var_variance, var_max, var_min

    if debug:
        var_mean, var_variance, var_max, var_min = variable_summary(graph_weight_1)

    # SORT POOLING LAYER
    nodes_size_list = list(nodes_size_list_train) + list(nodes_size_list_test)
    threshold_k = int(np.percentile(nodes_size_list, top_k))
    print("%s%% graphs have nodes less then %s." % (top_k, threshold_k))

    graph_conv_output_stored = tf.gather(graph_conv_output, tf.nn.top_k(Z_4[:, 0], node_size_pl).indices)
    graph_conv_output_top_k = tf.cond(tf.less(node_size_pl, threshold_k),
                                      lambda: tf.concat(axis=0,
                                                        values=[graph_conv_output_stored,
                                                                tf.zeros(dtype=tf.float32,
                                                                         shape=[threshold_k-node_size_pl,
                                                                                GRAPH_CONV_LAYER_CHANNEL*3])]),
                                      lambda: tf.slice(graph_conv_output_stored, begin=[0, 0], size=[threshold_k, -1]))

    # FLATTEN LAYER
    graph_conv_output_flatten = tf.reshape(graph_conv_output_top_k, shape=[1, GRAPH_CONV_LAYER_CHANNEL*3*threshold_k, 1])
    assert graph_conv_output_flatten.shape == [1, GRAPH_CONV_LAYER_CHANNEL*3*threshold_k, 1]

    # 1-D CONVOLUTION LAYER 1:
    # kernel = (filter_width, in_channel, out_channel)
    conv1d_kernel_1 = tf.Variable(tf.truncated_normal(shape=[CONV1D_1_FILTER_WIDTH, 1, CONV1D_1_OUTPUT], stddev=0.1, dtype=tf.float32))
    conv_1d_a = tf.nn.conv1d(graph_conv_output_flatten, conv1d_kernel_1, stride=CONV1D_1_FILTER_WIDTH, padding="VALID")
    assert conv_1d_a.shape == [1, threshold_k, CONV1D_1_OUTPUT]
    # 1-D CONVOLUTION LAYER 2:
    conv1d_kernel_2 = tf.Variable(tf.truncated_normal(shape=[CONV1D_2_FILTER_WIDTH, CONV1D_1_OUTPUT, CONV1D_2_OUTPUT], stddev=0.1, dtype=tf.float32))
    conv_1d_b = tf.nn.conv1d(conv_1d_a, conv1d_kernel_2, stride=1, padding="VALID")
    assert conv_1d_b.shape == [1, threshold_k - CONV1D_2_FILTER_WIDTH + 1, CONV1D_2_OUTPUT]
    conv_output_flatten = tf.layers.flatten(conv_1d_b)

    # DENSE LAYER
    weight_1 = tf.Variable(tf.truncated_normal(shape=[int(conv_output_flatten.shape[1]), DENSE_NODES], stddev=0.1))
    bias_1 = tf.Variable(tf.zeros(shape=[DENSE_NODES]))
    dense_z = tf.nn.relu(tf.matmul(conv_output_flatten, weight_1) + bias_1)
    if is_train == 1:
        dense_z = tf.layers.dropout(dense_z, DROP_OUTPUT_RATE)

    weight_2 = tf.Variable(tf.truncated_normal(shape=[DENSE_NODES, 2]))
    bias_2 = tf.Variable(tf.zeros(shape=[2]))
    pre_y = tf.matmul(dense_z, weight_2) + bias_2
    pos_score = tf.nn.softmax(pre_y)

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_pl, logits=pre_y))
    global_step = tf.Variable(0, trainable=False)

    train_data_size, test_data_size = X_train.shape[0], X_test.shape[0]

    # learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, train_data_size, LEARNING_RATE_DECAY, staircase=True)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
    # saver = tf.train.Saver()

    with tf.Session() as sess:
        print("start training gnn.")
        print("learning rate: %f. epoch: %d." % (learning_rate, epoch))
        start_t = time.time()
        sess.run(tf.global_variables_initializer())
        for epoch in range(epoch):
            batch_index = 0
            for _ in tqdm(range(train_data_size)):
                batch_index = batch_index + 1 if batch_index < train_data_size - 1 else 0
                feed_dict = {D_inverse_pl: D_inverse_train[batch_index],
                             A_tilde_pl: A_tilde_train[batch_index],
                             X_pl: X_train[batch_index],
                             Y_pl: Y_train[batch_index],
                             node_size_pl: nodes_size_list_train[batch_index],
                             is_train: 1
                             }
                loss_value, _, _ = sess.run([loss, train_op, global_step], feed_dict=feed_dict)

            train_acc = 0
            for i in range(train_data_size):
                feed_dict = {D_inverse_pl: D_inverse_train[i], A_tilde_pl: A_tilde_train[i],
                             X_pl: X_train[i], Y_pl: Y_train[i], node_size_pl: nodes_size_list_train[i], is_train: 0}
                pre_y_value = sess.run(pre_y, feed_dict=feed_dict)
                if np.argmax(pre_y_value, 1) == Y_train[i]:
                    train_acc += 1
            train_acc = train_acc / train_data_size

            test_acc, prediction, scores = 0, [], []
            for i in range(test_data_size):
                feed_dict = {D_inverse_pl: D_inverse_test[i], A_tilde_pl: A_tilde_test[i],
                             X_pl: X_test[i], Y_pl: Y_test[i], node_size_pl: nodes_size_list_test[i], is_train: 0}
                pre_y_value, pos_score_value = sess.run([pre_y, pos_score], feed_dict=feed_dict)
                prediction.append(np.argmax(pre_y_value, 1))
                scores.append(pos_score_value[0][1])
                if np.argmax(pre_y_value, 1) == Y_test[i]:
                    test_acc += 1
            test_acc = test_acc / test_data_size
            if debug:
                mean_value, var_value, max_value, min_value = sess.run([var_mean, var_variance, var_max, var_min], feed_dict=feed_dict)
                print("\t\tdebug: mean: %f, variance: %f, max: %f, min: %f." %
                      (mean_value, var_value, max_value, min_value))
            print("After %5s epoch, the loss is %f, training acc %f, test acc %f." % (epoch, loss_value, train_acc, test_acc))
            # saver.save(sess, os.path.join(MODEL_SAVE_PATH, data_name, MODEL_SAVE_NAME), global_step)
        end_t = time.time()
        print("time consumption: ", end_t - start_t)
    return test_acc, prediction, scores