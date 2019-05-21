from .pytorch_module import *
from .node2vec import *
def generate_graph_context_all_pairs(path, window_size):
    # generating graph context pairs
    all_pairs = []
    for k in range(len(path)):
        for i in range(len(path[k])):
            for j in range(i - window_size, i + window_size + 1):
                if i == j or j < 0 or j >= len(path[k]):
                    continue
                else:
                    all_pairs.append([path[k][i], path[k][j]])

    return np.array(all_pairs, dtype=np.int32)


def graph_context_batch_iter(all_pairs, batch_size):
    while True:
        start_idx = np.random.randint(0, len(all_pairs) - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch = np.zeros(batch_size, dtype=np.int32)
        labels = np.zeros((batch_size, 1), dtype=np.int32)
        batch[:] = all_pairs[batch_idx, 0]
        labels[:, 0] = all_pairs[batch_idx, 1]
        yield batch, labels


def construct_traget_neighbors(nx_G, X, FLAGS, mode='WAN'):
    # construct target neighbor feature matrix
    X_target = np.zeros(X.shape)
    nodes = nx_G.nodes()

    if mode == 'OWN':
        # autoencoder for reconstructing itself
        return X
    #
    elif mode == 'EMN':
        # autoencoder for reconstructing Elementwise Median Neighbor
        for node in nodes:
            neighbors = list(nx_G.neighbors(node))
            if len(neighbors) == 0:
                X_target[node] = X[node]
            else:
                temp = np.array(X[node])
                for n in neighbors:
                    if FLAGS.weighted:
                        # weighted sum
                        # temp = np.vstack((temp, X[n] * edgeWeight))
                        pass
                    else:
                        # temp  垂直的叠加
                        temp = np.vstack((temp, X[n]))
                # 计算特征的平均值
                temp = np.median(temp, axis=0)
                X_target[node] = temp
        return X_target
    elif mode == 'WAN':
        # autoencoder for reconstructing Weighted Average Neighbor
        for node in nodes:
            neighbors = list(nx_G.neighbors(node))
            if len(neighbors) == 0:
                X_target[node] = X[node]
            else:
                temp = np.array(X[node])
                for n in neighbors:
                    if FLAGS.weighted:
                        # weighted sum
                        # temp = np.vstack((temp, X[n] * edgeWeight))
                        pass
                    else:
                        temp = np.vstack((temp, X[n]))
                temp = np.mean(temp, axis=0)
                X_target[node] = temp
        return X_target


def main():
    config = read_param('./config.py')
    inputEdgeFile = config['inputEdgeFile']
    inputFeatureFile = config['inputFeatureFile']
    inputLabelFile = config['inputLabelFile']
    outputEmbedFile = config['outputEmbedFile']
    window_size = ['widow_size']

    # Read graph
    nx_G = read_graph(inputEdgeFile)

    # Perform random walks to generate graph context
    G = Graph_node2vec(nx_G,config['p'], config['q'])
    G.preprocess_transition_probs()
    walks = G.simulate_walks(config['num_walks'], config['walk_length'])

    # Read features
    print('reading features...')
    X = read_feature(inputFeatureFile)

    print('generating graph context pairs...')
    start_time = time.time()
    # 每一个节点 randomwalk 的node对
    all_pairs = generate_graph_context_all_pairs(walks, window_size)
    end_time = time.time()
    print('time consumed for constructing graph context: %.2f' % (end_time - start_time))

    nodes = nx_G.nodes()
    X_target = construct_traget_neighbors(nx_G, X, FLAGS, mode='WAN')

    # Total number nodes
    N = len(nodes)
    dims = FLAGS.dimensions

    config = Config()
    config.struct[0] = FLAGS.feaDims
    config.struct[-1] = FLAGS.dimensions
    model = Model(config, N, dims, X_target)

    init = tf.global_variables_initializer()
    sess = tf.Session(config=config_tf)
    sess.run(init)

    batch_size = config.batch_size
    max_iters = config.max_iters

    idx = 0
    print_every_k_iterations = 1000
    start = time.time()

    loss_sg = 0
    loss_ae = 0

    for iter_cnt in range(max_iters):
        idx += 1

        batch_index, batch_labels = next(graph_context_batch_iter(all_pairs, batch_size))

        # train for autoencoder model
        start_idx = np.random.randint(0, N - batch_size)
        batch_idx = np.array(range(start_idx, start_idx + batch_size))
        batch_idx = np.random.permutation(batch_idx)
        batch_X = X[batch_idx]
        feed_dict = {model.X: batch_X, model.inputs: batch_idx}
        _, loss_ae_value = sess.run([model.train_opt_ae, model.loss_ae], feed_dict=feed_dict)
        loss_ae += loss_ae_value

        # train for skip-gram model
        batch_X = X[batch_index]
        feed_dict = {model.X: batch_X, model.labels: batch_labels}
        _, loss_sg_value = sess.run([model.train_opt_sg, model.loss_sg], feed_dict=feed_dict)
        loss_sg += loss_sg_value

        if idx % print_every_k_iterations == 0:
            end = time.time()
            print('iterations: %d' % idx + ', time elapsed: %.2f, ' % (end - start), end='')
            total_loss = loss_sg / idx + loss_ae / idx
            print('loss: %.2f, ' % total_loss, end='')

            y = read_label(inputLabelFile)
            embedding_result = sess.run(model.Y, feed_dict={model.X: X})
            macro_f1, micro_f1 = multiclass_node_classification_eval(embedding_result, y, 0.7)
            print('[macro_f1 = %.4f, micro_f1 = %.4f]' % (macro_f1, micro_f1))

    print('optimization finished...')
    y = read_label(inputLabelFile)
    embedding_result = sess.run(model.Y, feed_dict={model.X: X})
    print('repeat 10 times for node classification with random split...')
    node_classification_F1(embedding_result, y)
    print('saving embedding result...')
    write_embedding(embedding_result, outputEmbedFile)


if __name__ == '__main__':
    main()