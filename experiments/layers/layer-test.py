'''
Created on May 20, 2017

@author: stelzner
'''
from TFSPN.src.tfspn.layers import *
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def plot_marginals(points, probs):
    plt.scatter(points[:, 0], points[:, 1], c=probs,
                cmap=cm.get_cmap("inferno"), marker='o',
                edgecolors='none')


def plot_classification(points, labels, flags='o'):
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'r' + flags)
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'b' + flags)


def plot_grid_prediction(cond, x, testdata=None):
    sample = np.arange(-15, 15, 0.25)
    grid = np.asarray(list(itertools.product(sample, sample)))
    grid_with_labels = np.zeros((120 * 120, 3))
    grid_with_labels[:, :-1] = grid
    for i in range(120 * 120):
        grid_with_labels[i, -1] = np.exp(cond.eval(feed_dict={x: grid_with_labels[i:i+1,:]}))

    plot_marginals(grid, grid_with_labels[:, -1])
    if testdata is not None:
        plot_classification(testdata[:, 0:2], testdata[:, 2])


if __name__ == '__main__':
    name = "twospirals"
    data = np.loadtxt("standard/" + name + ".csv", delimiter=",")

    x = data[:, :2]
    y = data[:, 2]

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.3, random_state=1337)

    traindata = np.c_[train_x, train_y]
    testdata = np.c_[test_x, test_y]

    num_prod_nodes = 20

    gaussian_features = []
    prod_structure = np.zeros((num_prod_nodes, num_prod_nodes * 3))
    for i in range(num_prod_nodes):
        gaussian_features += [0, 1]
        prod_structure[i][i * 2:i * 2 + 2] = 1
        prod_structure[i][num_prod_nodes * 2 + i] = 1

    gaussian_layer = GaussianLayer(gaussian_features)
    bernoulli_layer = BernoulliLayer([2] * num_prod_nodes)
    prod_layer = ProductLayer([gaussian_layer, bernoulli_layer], np.transpose(prod_structure))
    sum_layer = SumLayer([prod_layer], np.ones((num_prod_nodes, 1)))

    # Testing layer-node translation
    sum_layer.to_nodes()
    translation = NodesToLayers(sum_layer.nodes[0])
    print('old size:', sum_layer.network_size())
    sum_layer = translation.root_layer
    print('valid?', sum_layer.validate())
    print('new size:', sum_layer.network_size())

    input_placeholder = tf.placeholder(tf.float64, shape=[None, 3])

    joined_prob = sum_layer.value(input_placeholder)
    marginalized_prob = sum_layer.value(input_placeholder, [False, False, True])
    conditional_prob = joined_prob - marginalized_prob
    loss = -1 * tf.reduce_sum(conditional_prob, 0)

    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(201):
            for i in range(0, 1400, 50):
                tr = sess.run([train_op], feed_dict={input_placeholder: traindata[i: i+50,:]})
            if epoch % 50 == 0:
                cond = loss.eval(feed_dict={input_placeholder: traindata})
                print('Loss after epoch ' + str(epoch) + ': ')
                print(cond)

        plot_grid_prediction(conditional_prob, input_placeholder, testdata=testdata)
        plt.show()



