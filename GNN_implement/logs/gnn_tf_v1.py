import argparse
import os.path
import glob
import numpy as np
import time
import tensorflow as tf

EPOCH = 100

parser = argparse.ArgumentParser()
args = parser.parse_args()


def load_networks():
    print("load data...")
    file_list = []
    file_glob_pattern = os.path.join("graph_data", "mutag", "mutag*.graph")
    file_list.extend(glob.glob(file_glob_pattern))

    edges_set, labels_set = {}, {}
    node_size = 0
    for file in file_list:
        base_name = os.path.basename(file)
        edges_set[base_name] = []
        with open(file, "r") as f:
            line, read = f.readline(), False
            while line:
                if line.startswith("#c - Class"):
                    labels_set[base_name] = (int(f.readline().strip()))
                    break
                if read:
                    edges_set[base_name].append([int(x) for x in line.strip().split(",")[:2]])
                if line.startswith("#e - edge labels"):
                    read = True
                line = f.readline()
        max_temp = max(max(np.array(edges_set[base_name])[:, 0]), max(np.array(edges_set[base_name])[:, 1]))
        node_size = node_size if node_size > max_temp else max_temp

    A, Y, count = [], [], 0
    for key, value in edges_set.items():
        A.append(np.zeros([node_size, node_size], dtype=np.float32))
        for edge in value:
            A[count][edge[0] - 1][edge[1] - 1] = 1.
        Y.append([labels_set[key]])
        count += 1
    A, Y = np.array(A), np.array(Y)
    Y = np.where(Y == -1, 0, 1)
    print("\tpositive examples: %d, negative examples: %d.", np.sum(Y == 0), np.sum(Y == 1))
    print("\tX.shape: ", A.shape)
    print("\tY.shape: ", Y.shape)

    # get A_tilde
    A_tilde = np.eye(node_size) + A

    # get D_inverse
    D = np.sum(A_tilde, axis=2)
    D = np.array([np.diag(x) for x in D])
    D_inverse = np.linalg.inv(D)

    # get graph_size_list
    graph_size_list = [x.shape[1] for x in A_tilde]
    return D_inverse, A_tilde, Y, node_size, graph_size_list


def create_embedding_or_attribution(attribution, dimension, graph_size_list):
    """
    :param attribution: 
    :param dimension: 
    :param graph_size_list: 
    :return: X
    """
    if attribution == "label":
        return np.array([np.eye(x) for x in graph_size_list])
    if attribution == "n2v":
        pass


def split_train_test(D_inverse, A_tilde, X, Y, rate):
    state = np.random.get_state()
    np.random.shuffle(D_inverse)
    np.random.set_state(state)
    np.random.shuffle(A_tilde)
    np.random.set_state(state)
    np.random.shuffle(X)
    np.random.set_state(state)
    np.random.shuffle(Y)
    data_size = Y.shape[0]
    training_set_size, test_set_size = int(data_size * (1 - rate)), int(data_size * rate)
    D_inverse_train, D_inverse_test = D_inverse[: training_set_size], D_inverse[training_set_size: ]
    A_tilde_train, A_tilde_test = A_tilde[: training_set_size], A_tilde[training_set_size: ]
    X_train, X_test = X[: training_set_size], X[training_set_size: ]
    Y_train, Y_test = Y[: training_set_size], Y[training_set_size: ]
    return D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test


def GNN(X_train, D_inverse_train, A_tilde_train, Y_train, node_size, top_k, initial_channels,
        X_test, D_inverse_test, A_tilde_test, Y_test):
    # placeholder
    D_inverse_pl = tf.placeholder(dtype=tf.float32, shape=[node_size, node_size])
    A_tilde_pl = tf.placeholder(dtype=tf.float32, shape=[node_size, node_size])
    X_pl = tf.placeholder(dtype=tf.float32, shape=[node_size, initial_channels])
    Y_pl = tf.placeholder(dtype=tf.int32, shape=[1])

    # graph convolution layer
    graph_weight_1 = tf.Variable(tf.truncated_normal(shape=[28, 32], stddev=0.1, dtype=tf.float32))
    graph_weight_2 = tf.Variable(tf.truncated_normal(shape=[32, 32], stddev=0.1, dtype=tf.float32))
    graph_weight_3 = tf.Variable(tf.truncated_normal(shape=[32, 32], stddev=0.1, dtype=tf.float32))
    graph_weight_4 = tf.Variable(tf.truncated_normal(shape=[32, 1], stddev=0.1, dtype=tf.float32))

    # forward pass
    # graph convolution layer
    gl_1_XxW = tf.matmul(X_pl, graph_weight_1)                  # shape=(node_size, 32)
    gl_1_AxXxW = tf.matmul(A_tilde_pl, gl_1_XxW)                # shape=(node_size, 32)
    Z_1 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_1_AxXxW))       # shape=(node_size, 32)
    # graph convolution layer
    gl_2_XxW = tf.matmul(Z_1, graph_weight_2)                   # shape=(node_size, 32)
    gl_2_AxXxW = tf.matmul(A_tilde_pl, gl_2_XxW)                # shape=(node_size, 32)
    Z_2 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_2_AxXxW))       # shape=(node_size, 32)
    # graph convolution layer
    gl_3_XxW = tf.matmul(Z_2, graph_weight_3)                   # shape=(node_size, 32)
    gl_3_AxXxW = tf.matmul(A_tilde_pl, gl_3_XxW)                # shape=(node_size, 32)
    Z_3 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_3_AxXxW))       # shape=(node_size, 32)
    # graph convolution layer
    gl_4_XxW = tf.matmul(Z_3, graph_weight_4)                   # shape=(node_size, 1)
    gl_4_AxXxW = tf.matmul(A_tilde_pl, gl_4_XxW)                # shape=(node_size, 1)
    Z_4 = tf.nn.tanh(tf.matmul(D_inverse_pl, gl_4_AxXxW))       # shape=(node_size, 1)
    graph_conv_output = tf.concat([Z_1, Z_2, Z_3], axis=1)      # shape=(node_size, 32+32+32)
    assert graph_conv_output.shape == [node_size, 32*3]

    # sortpooling layer
    graph_conv_output_stored = tf.gather(graph_conv_output, tf.nn.top_k(-Z_4[:, 0], node_size).indices)
    graph_conv_output_top_k = tf.slice(graph_conv_output_stored, begin=[0, 0], size=[top_k, -1]) # shape=(k, 32+32+32)
    assert graph_conv_output_top_k.shape == [top_k, 32*3]
    graph_conv_output_flatten = tf.reshape(graph_conv_output_top_k, shape=[1, -1, 1])
                                                                # shape=(1, (32+32+32)*25, 1)
    assert graph_conv_output_flatten.shape == [1, 96*top_k, 1]

    # 1-D convolution layer
    # (filter_width, in_channel, out_channel)
    width = 3 * 32
    conv1d_kernel_1 = tf.Variable(tf.truncated_normal(shape=[width, 1, 16], stddev=0.1, dtype=tf.float32))
    conv_1d_a = tf.nn.conv1d(graph_conv_output_flatten, conv1d_kernel_1, stride=width, padding="VALID")
                                                                # shape=(1, k, 16)
    assert conv_1d_a.shape == [1, top_k, 16]
    conv1d_kernel_2 = tf.Variable(tf.truncated_normal(shape=[5, 16, 32], stddev=0.1, dtype=tf.float32))
    conv_1d_b = tf.nn.conv1d(conv_1d_a, conv1d_kernel_2, stride=1, padding="VALID")     # shape=(1, x, 32)
    assert conv_1d_b.shape == [1, top_k - 5 + 1, 32]
    conv_output_flatten = tf.layers.flatten(conv_1d_b)

    # dense layer
    weight_1 = tf.Variable(tf.truncated_normal(shape=[int(conv_output_flatten.shape[1]), 128], stddev=0.1))
    bias_1 = tf.Variable(tf.zeros(shape=[128]))
    dense_z = tf.nn.relu(tf.matmul(conv_output_flatten, weight_1) + bias_1)
    drop_dense_z = tf.layers.dropout(dense_z, 0.7)

    weight_2 = tf.Variable(tf.truncated_normal(shape=[128, 2]))
    bias_2 = tf.Variable(tf.zeros(shape=[2]))
    pre_y = tf.matmul(drop_dense_z, weight_2) + bias_2

    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=Y_pl, logits=pre_y))
    train_op = tf.train.AdamOptimizer(0.0001).minimize(loss)

    train_data_size = X_train.shape[0]
    test_data_size = X_test.shape[0]

    # debug
    graph_weight_1_mean = tf.reduce_mean(graph_weight_1)
    graph_weight_1_variance = tf.square(tf.reduce_mean(tf.square(graph_weight_1 - tf.reduce_mean(graph_weight_1))))
    graph_weight_1_max = tf.reduce_max(graph_weight_1)
    graph_weight_1_min = tf.reduce_min(graph_weight_1)

    with tf.Session() as sess:
        print("\nstart training gnn.")
        start_t = time.time()
        sess.run(tf.global_variables_initializer())
        batch_index = 0
        for step in range(EPOCH * train_data_size):
            batch_index = batch_index + 1 if batch_index < train_data_size - 1 else 0
            feed_dict = {D_inverse_pl: D_inverse_train[batch_index],
                         A_tilde_pl: A_tilde_train[batch_index],
                         X_pl: X_train[batch_index],
                         Y_pl: Y_train[batch_index]
                         }
            loss_value, _ = sess.run([loss, train_op], feed_dict=feed_dict)
            if step % 1000 == 0:
                train_acc = 0
                for i in range(train_data_size):
                    feed_dict = {D_inverse_pl: D_inverse_train[i],
                                 A_tilde_pl: A_tilde_train[i],
                                 X_pl: X_train[i],
                                 Y_pl: Y_train[i]}
                    pre_y_value = sess.run(pre_y, feed_dict=feed_dict)
                    if np.argmax(pre_y_value, 1) == Y_train[i]:
                        train_acc += 1
                train_acc = train_acc / train_data_size

                test_acc = 0
                for i in range(test_data_size):
                    feed_dict = {D_inverse_pl: D_inverse_test[i],
                                 A_tilde_pl: A_tilde_test[i],
                                 X_pl: X_test[i],
                                 Y_pl: Y_test[i]}
                    pre_y_value = sess.run(pre_y, feed_dict=feed_dict)
                    if np.argmax(pre_y_value, 1) == Y_test[i]:
                        test_acc += 1
                test_acc = test_acc / test_data_size

                # mean_value, var_value, max_value, min_value = sess.run([graph_weight_1_mean,
                #                                                         graph_weight_1_variance,
                #                                                         graph_weight_1_max,
                #                                                         graph_weight_1_min],
                #                                                        feed_dict=feed_dict)
                # print("\tdebug: mean: %f, variance: %f, max: %f, min: %f" %
                #       (mean_value, var_value, max_value, min_value))

                print("After %5s step, the loss is %f, training acc %f, test acc %f."
                      % (step, loss_value, train_acc, test_acc))
        end_t = time.time()
        print("time consumption: ", end_t - start_t)


if __name__ == "__main__":
    D_inverse, A_tilde, Y, node_size, graph_size_list = load_networks()
    X = create_embedding_or_attribution("label", 0, graph_size_list)
    D_inverse_train, D_inverse_test, A_tilde_train, A_tilde_test, X_train, X_test, Y_train, Y_test = \
        split_train_test(D_inverse, A_tilde, X, Y, 0.1)
    GNN(X_train, D_inverse_train, A_tilde_train, Y_train, node_size, 25, 28,
        X_test, D_inverse_test, A_tilde_test, Y_test)

    # GNN(X, D_inverse, A_tilde, Y, node_size, 25, 28, None, None, None, None)
