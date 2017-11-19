'''
Created on May 20, 2017

@author: stelzner
'''
from tfspn.layers import SumLayer, ProductLayer, GaussianLayer, BernoulliLayer, to_layers, discriminative_cost
import tfspn.tfspn as nodes
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import itertools
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation


def plot_marginals(points, probs):
    plt.scatter(points[:, 0], points[:, 1], c=probs,
                cmap=cm.get_cmap("inferno"), marker='o',
                edgecolors='none')


def plot_classification(fig, points, labels, flags='o'):
    plt.plot(points[labels == 0, 0], points[labels == 0, 1], 'r' + flags)
    plt.plot(points[labels == 1, 0], points[labels == 1, 1], 'b' + flags)


def plot_disc_grid(cond, x, fig, testdata=None):
    sample = np.arange(-15, 15, 0.20)
    grid = np.asarray(list(itertools.product(sample, sample)))
    grid_with_labels = np.zeros((len(grid), 4))
    grid_with_labels[:, 0:2] = grid
    grid_with_labels[:, 3] = 1
    grid_with_labels[:, 2] = np.exp(cond.eval(feed_dict={x: grid_with_labels}))[:, 0]

    plot_marginals(grid, grid_with_labels[:, -1])
    if testdata is not None:
        plot_classification(fig, testdata[:, 0:2], testdata[:, 2])


def plot_grid_prediction(cond, x, fig, testdata=None):
    sample = np.arange(-15, 15, 0.20)
    grid = np.asarray(list(itertools.product(sample, sample)))
    grid_with_labels = np.zeros((len(grid), 3))
    grid_with_labels[:, :-1] = grid

    grid_with_labels[:, -1] = np.exp(cond.eval(feed_dict={x: grid_with_labels}))[:, 0]

    plot_marginals(grid, grid_with_labels[:, -1])
    if testdata is not None:
        plot_classification(fig, testdata[:, 0:2], testdata[:, 2])


def build_generator():
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
    return sum_layer

def build_discriminator():
    num_prod_nodes = 20
    gaussian_features = []
    prod_structure = np.zeros((num_prod_nodes, num_prod_nodes * 4))
    for i in range(num_prod_nodes):
        gaussian_features += [0, 1]
        prod_structure[i][i * 2:i * 2 + 2] = 1
        prod_structure[i][num_prod_nodes * 2 + i] = 1
        prod_structure[i][num_prod_nodes * 3 + i] = 1

    gaussian_layer = GaussianLayer(gaussian_features)
    bernoulli_layer = BernoulliLayer(([2] * num_prod_nodes) + ([3] * num_prod_nodes))
    prod_layer = ProductLayer([gaussian_layer, bernoulli_layer], np.transpose(prod_structure))
    sum_layer = SumLayer([prod_layer], np.ones((num_prod_nodes, 1)))
    return sum_layer

if __name__ == '__main__':
    name = "twospirals"
    data = np.loadtxt("standard/" + name + ".csv", delimiter=",")

    x_ = data[:, :2]
    y_ = data[:, 2]

    train_x, test_x, train_y, test_y = train_test_split(x_, y_, test_size=0.3, random_state=1337)

    traindata = np.c_[train_x, train_y]
    testdata = np.c_[test_x, test_y]

    generator = build_generator()
    discriminator = build_discriminator()
    print(generator.validate())
    print(discriminator.validate())
    x = tf.placeholder(tf.float64, shape=[None, 3])
    disc_x = tf.placeholder(tf.float64, shape=[None, 4])

    with tf.name_scope('SPN') as scope:

        to_nodes = False
        if not to_nodes:
            zeros = tf.zeros_like(x)
            samples_disc, sample_probs_disc = generator.map_value(zeros, [False] * 3, sample=True)
            samples_disc = tf.concat((samples_disc[:, 0, :], tf.ones_like(zeros[:, 0:1])), axis=1)
            samples_gen, sample_probs_gen = generator.map_value(zeros, [False] * 3, sample=True)
            samples_gen = tf.concat((samples_gen[:, 0, :], tf.ones_like(zeros[:, 0:1])), axis=1)
            real_data = tf.concat((x, zeros[:, 0:1]), axis=1)

            disc_input = tf.concat((samples_disc, real_data), axis=0)
            disc_loss = discriminative_cost(disc_input, [False, False, False, True], discriminator)
            gen_loss = -1 * discriminative_cost(samples_gen, [False, False, False, True], discriminator)

            # disc_joined_real = discriminator.value(real_data)
            # disc_marginal_real = discriminator.value(real_data, [False, False, False, True])
            # disc_conditional_realness_real = disc_joined_real - disc_marginal_real
            # disc_loss = -1 * disc_conditional
            disc_joint = discriminator.value(disc_x)
            disc_marginal = discriminator.value(disc_x, [False, False, False, True])

            joint = generator.value(x)
            marginal = generator.value(x, [False, False, True])
            conditional = tf.exp(joint - marginal)
            loss = -1 * tf.reduce_sum(conditional, 0)
        else:
            generator.to_nodes()
            joint = generator.nodes[0]
            joint.initTFSharedData(x)
            joint.initTf()
            marginal = joint.marginalizeOut([2])
            marginal.initTFSharedData(x)
            marginal.initTf()
            loss = nodes.DiscriminativeCost(joint, marginal)
            conditional = tf.exp(joint.value - marginal.value)
            print(joint)
            print(marginal)
            # to_layers(joint)
            joint = joint.value
            marginal = marginal.value

        #train_op = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
        train_gen = tf.train.AdamOptimizer(learning_rate=0.1).minimize(gen_loss, var_list=generator.var_list())
        train_disc = tf.train.AdamOptimizer(learning_rate=0.1).minimize(disc_loss, var_list=discriminator.var_list())
        FFMpegWriter = animation.writers['ffmpeg']
        metadata = dict(title='Movie Test', artist='Matplotlib',
                        comment='Movie support!')
        writer = FFMpegWriter(fps=30, metadata=metadata, bitrate=2000)

        with tf.Session() as sess:
            fig = plt.figure(figsize=(7, 7), dpi=400)
            axes = plt.gca()
            axes.set_xlim([-15, 15])
            axes.set_ylim([-15, 15])
            with writer.saving(fig, "grad_desc.mp4", 400):
                sess.run(tf.global_variables_initializer())
                for epoch in range(500):

                    if epoch % 50 == 0:
                        cur_gen_loss = gen_loss.eval(feed_dict={x: traindata})
                        cur_disc_loss = disc_loss.eval(feed_dict={x: traindata})
                        print('Loss after epoch ' + str(epoch) + ': ')
                        print('Generator', cur_gen_loss)
                        print('Discriminator', cur_disc_loss)
                        test_cond = conditional.eval(feed_dict={x: testdata})

                        # plt.show()
                        # print('joint')
                        # print(joint.eval(feed_dict={input_placeholder: testdata[:10,:]}))
                        # print('marginal')
                        # print(marginal.eval(feed_dict={input_placeholder: testdata[:10, :]}))
                        # print('conditional')
                        # print(test_cond[:10])
                        print('ACC:', (test_cond > 0.5).sum() / len(testdata))
                        # print(disc_input.eval(feed_dict={x: traindata[0:100, :]}))
                        # plot_disc_grid(disc_joint - disc_marginal, disc_x, fig, testdata=testdata)
                        # plt.show()
                    for i in range(0, 1400, 100):
                        sample_vals = samples_gen.eval(feed_dict={x: testdata})
                        plot_classification(plt, testdata, testdata[:, 2])
                        plot_classification(plt, sample_vals[:, :], sample_vals[:, 2], flags='x')
                        writer.save_fig()
                        sess.run([train_gen], feed_dict={x: traindata[i: i + 100, :]})
                        sess.run([train_disc], feed_dict={x: traindata[i: i + 100, :]})

                # predictions = predict(spn, testdata, 2)
            # print('MAP accuracy : ', accuracy_score(test_y, predictions))




