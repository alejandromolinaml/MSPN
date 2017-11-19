'''
Created on May 23, 2017

@author: stelzner
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# import experiments.mnist.augmenting_mnist as aug
import time
import numpy as np
from tfspn.layers import *
import tfspn.tfspn as spn
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
import warnings
from tensorflow.python import debug as tf_debug
from tensorflow.python.debug.lib import debug_data as debug_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def plot_MNIST(plot, pixels, name="test"):
    pixels = np.array(pixels)
    pixels = pixels.reshape((28, 28))
    return plot.imshow(pixels, cmap='gray', interpolation='none')
    # plt.savefig('images/' + name + '.png')
    # plt.show()


def get_structure_matrix_prods(rows, cols):
    if rows < cols:
        res, new_rows, new_cols = get_structure_matrix_prods(cols, rows)
        return np.transpose(res, axes=(1, 0, 2, 4, 3, 5)), new_cols, new_rows
    new_rows = (rows + 1) // 2
    res = np.zeros((rows, cols, 2, new_rows, cols, 4))
    for y in range(cols):
        for x in range(new_rows):
            res[x * 2, y, 0, x, y, 0] = 1
            res[x * 2, y, 0, x, y, 1] = 1
            res[x * 2, y, 1, x, y, 2] = 1
            res[x * 2, y, 1, x, y, 3] = 1
            if x * 2 + 1 < rows:
                res[x * 2 + 1, y, 0, x, y, 0] = 1
                res[x * 2 + 1, y, 1, x, y, 1] = 1
                res[x * 2 + 1, y, 0, x, y, 2] = 1
                res[x * 2 + 1, y, 1, x, y, 3] = 1
    return res, new_rows, cols


def create_merging_layers(input_layer, rows, cols):
    prod_structure, new_rows, new_cols = get_structure_matrix_prods(rows, cols)
    prod_structure_reshaped = np.reshape(prod_structure, (rows * cols * 2, new_rows * new_cols * 4), order='F')
    prod_layer = ProductLayer([input_layer], prod_structure_reshaped)

    sum_structure = np.zeros((new_rows, new_cols, 4, new_rows, new_cols, 2))
    for x in range(new_rows):
        for y in range(new_cols):
            sum_structure[x, y, :, x, y, :] = 1
    sum_structure_reshaped = np.reshape(sum_structure, (new_rows * new_cols * 4, new_rows * new_cols * 2), order='F')
    sum_layer = SumLayer([prod_layer], sum_structure_reshaped)
    return sum_layer, new_rows, new_cols


def build_merging_network(input_layer, rows, cols):
    while rows > 1 or cols > 1:
        input_layer, rows, cols = create_merging_layers(input_layer, rows, cols)
        # print(rows, cols)
    return SumLayer([input_layer], np.ones((2, 1)))


def build_merging_model(feature, rows, cols):
    coarseness = 4
    coarse_rows = rows // coarseness
    coarse_cols = cols // coarseness
    features = list(range(rows * cols)) * 2
    gaussian_layer = GaussianLayer(features)
    prod_structure = np.zeros((rows*cols*2, coarse_rows*coarse_cols*2))
    for i in range(rows*cols):
        for j in range(coarse_rows*coarse_cols):
            cur_row = i // cols
            cur_col = i % cols
            if (j // coarse_cols) * coarseness <= cur_row < ((j // coarse_cols) + 1) * coarseness:
                if (j % coarse_cols) * coarseness <= cur_col < ((j % coarse_cols) + 1) * coarseness:
                    prod_structure[i][j] = 1

    prod_structure[rows*cols:, coarse_rows*coarse_cols:] = prod_structure[:rows*cols, :coarse_rows*coarse_cols]
    product_layer = ProductLayer([gaussian_layer], prod_structure)
    complete_layer = build_merging_network(product_layer, coarse_rows, coarse_cols)
    discrete = DiscreteLayer([feature], values=[1.0])
    res = ProductLayer([complete_layer, discrete], np.ones((2, 1)))
    # print('Validate:', res.validate())
    return res


def build_mnist_model():
    spns = []
    for i in range(10):
        spns.append(build_merging_model(28 * 28 + i, 28, 28))
    return SumLayer(spns, np.ones((10, 1)))


def test_spn(logits, x, features, labels):
    testsize = features.shape[0]
    testset = np.concatenate((features, np.zeros((testsize, labels.shape[1]))), 1)
    conds = np.zeros((testsize, labels.shape[1]))
    for i in range(labels.shape[1]):
        testset[:, features.shape[1] + i] = 1
        conds[:, i] = logits.eval(feed_dict={x: testset})[:, 0]
        testset[:, features.shape[1] + i] = 0
    preds = np.argmax(conds, 1)
    truth = np.argmax(labels, 1)
    score = 0
    for i in range(testsize):
        if preds[i] == truth[i]:
            score += 1
    return score / testsize


class Timer:
    def __init__(self):
        self.t = time.time()

    def checkpoint(self):
        interval = time.time() - self.t
        self.t = time.time()
        return interval


def map_label_to_image(sess, layer, x):
    evidence = np.zeros((10, 28 * 28 + 10))
    for i in range(10):
        evidence[i, 28 * 28 + i] = 1

    map_v_t, map_p_t = layer.map_value(x, ([False] * (28 * 28)) + ([True] * 10))
    v, p = sess.run([map_v_t, map_p_t], feed_dict={x: evidence})
    fig = plt.figure()

    for i in range(10):
        print(p[i])
        plot_MNIST(v[i, 0, :28 * 28], str(i))


def map_complete_half_image(sess, layer, x):
    batch_size = 9
    evidence = mnist.test.next_batch(batch_size)
    real_labels = np.argmax(evidence[1], 1)
    evidence = np.concatenate((evidence[0], evidence[1]), 1)

    mask = np.concatenate(([False] * (28 * 14), [True] * (28 * 14), [False] * 10), 0)

    for i in range(len(mask)):
        if not mask[i]:
            evidence[:, i] = 0

    map_v, map_p = layer.map_value(x, mask)
    v = sess.run([map_v], feed_dict={x: evidence})[0]
    # print('real labels', real_labels)
    fig, axes = plt.subplots(3, 3)
    for i in range(batch_size):
        estimated_label = np.argmax(v[i, 0, 28 * 28:], 0)
        plot = axes[i % 3, i // 3]
        plot.axhline(y=13.5, xmin=0, xmax=28, color='r')
        plot.set_title(str(real_labels[i]) + ' -> ' + str(estimated_label))
        plot.axes.get_xaxis().set_visible(False)
        plot.axes.get_yaxis().set_visible(False)
        plot_MNIST(plot, v[i, 0, :28 * 28], str(i))
    plt.savefig('images/all.png')


def sample(sess, layer, x):
    sampler, p = layer.map_value(x, ([False] * (28 * 28)) + [True] * 10, True)

    for j in range(5):
        fig, axes = plt.subplots(4, 5)
        evidence = np.zeros((20, 28 * 28 + 10))
        for i in range(10):
            evidence[2 * i: 2 * i + 2, 28 * 28 + i] = 1
        samples = sess.run([sampler], feed_dict={x: evidence})[0]
        for i in range(20):
            print(np.argmax(samples[i, 0, 28 * 28:], 0))
            plot = axes[i % 4, i // 4]
            plot_MNIST(plot, samples[i, 0, :28 * 28])
        plt.savefig('images/samples' + str(j) + '.png')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float64)
    return tf.Variable(initial)


class Discriminator:
    def __init__(self):
        self.D_W1 = weight_variable([794, 128])
        self.D_b1 = tf.Variable(tf.zeros(shape=[128], dtype=tf.float64), name='D_b1', dtype=tf.float64)

        self.D_W2 = weight_variable([128, 1])
        self.D_b2 = tf.Variable(tf.zeros(shape=[1], dtype=tf.float64), name='D_b2', dtype=tf.float64)
        self.var_list = [self.D_W1, self.D_b1, self.D_W2, self.D_b2]

    def feed(self, inp):
        D_h1 = tf.nn.sigmoid(tf.matmul(inp, self.D_W1) + self.D_b1)
        D_logit = tf.matmul(D_h1, self.D_W2) + self.D_b2
        D_prob = tf.nn.sigmoid(D_logit)
        return D_prob


def has_nan(datum, tensor):
    _ = datum  # Datum metadata is unused in this predicate.

    if tensor is None:
        # Uninitialized tensor doesn't have bad numerical values.
        # Also return False for data types that cannot be represented as numpy
        # arrays.
        return False
    elif (np.issubdtype(tensor.dtype, np.float) or
          np.issubdtype(tensor.dtype, np.complex) or
          np.issubdtype(tensor.dtype, np.int32)):
        return np.any(np.isnan(tensor))
    else:
        return False


def adversarial_training(sess, spn):
    batch_size = 10
    discriminator = Discriminator()

    real_data_placeholder = tf.placeholder(dtype=tf.float64, shape=[None, 28 * 28 + 10])
    zero_input = tf.zeros((batch_size, 28 * 28 + 10), dtype=tf.float64)
    fake_data, sample_probs = spn.map_value(zero_input, [False] * (28 * 28 + 10), True)
    d_real = discriminator.feed(real_data_placeholder)
    d_fake = discriminator.feed(fake_data[:, 0, :])
    disc_loss = -1 * (tf.reduce_sum(tf.log(d_real)) + tf.reduce_sum(tf.log(1 - d_fake)))
    gen_loss = -1 * tf.reduce_sum(tf.log(d_fake + 0.01))
    conditional_prob = spn.value(real_data_placeholder) - spn.value(real_data_placeholder,
                                                                    ([False] * (28 * 28)) + [True] * 10)
    disc_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(disc_loss, var_list=discriminator.var_list)
    gen_opt = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(gen_loss, var_list=spn.var_list())
    sess.run(tf.global_variables_initializer())
    print('adversarial loss computed, starting training')
    for i in range(1000):
        real_data = mnist.train.next_batch(batch_size)
        real_data = np.concatenate((real_data[0], real_data[1]), axis=1)
        sess.run(disc_opt, feed_dict={real_data_placeholder: real_data})
        sess.run(gen_opt, feed_dict={real_data_placeholder: real_data})
        if i % 1 == 0:
            testdata = mnist.test.next_batch(batch_size)
            acc = test_spn(conditional_prob, real_data_placeholder, testdata[0], testdata[1])
            testdata = np.concatenate((testdata[0], testdata[1]), axis=1)
            print(i, acc)
            print('disc loss', sess.run(disc_loss, feed_dict={real_data_placeholder: testdata}))
            print('gen loss', sess.run(gen_loss, feed_dict={real_data_placeholder: testdata}))





def train_mnist_model(to_nodes=False, adversarial=False, restore=None, save=None, train=0, debug=False):
    logfile = open('mnist_test.log', 'w')
    timer = Timer()
    root_layer = build_mnist_model()

    print('Layer model built in ' + repr(timer.checkpoint()) + ' seconds')
    num_layers, num_nodes, num_variables = root_layer.network_size()
    print('Model size:', num_layers, 'layers,', num_nodes, 'nodes,', num_variables, 'variables')

    if to_nodes:
        root_layer.to_nodes()
        #joint = root_layer.nodes[0]
        #joint.initTFSharedData(x)
        #joint.initTf()
        #marginal = joint.marginalizeOut(list(range(28*28, 28*28 + 10)))
        #marginal.initTFSharedData(x)
        #marginal.initTf()
        #cost = spn.DiscriminativeCost(joint, marginal)
        print('Node model built in ' + repr(timer.checkpoint()) + ' seconds')
        print('old size', root_layer.network_size())
        print('original valid?', root_layer.validate())
        root_layer = translate_nodes_to_layers(root_layer.nodes[0])
        for child in root_layer.inputs:
            print('translated layer model valid?', child.validate(True))
        print('new size', root_layer.network_size())
        print('Node -> Layer translation completed in', repr(timer.checkpoint()), 'seconds')

    if adversarial:

        with tf.Session() as sess:
            if debug:
                sess = tf_debug.LocalCLIDebugWrapperSession(sess)
                sess.add_tensor_filter("has_nan", has_nan)
            adversarial_training(sess, root_layer)
        return

    x = tf.placeholder(dtype=tf.float64, shape=[None, 28 * 28 + 10])
    joined_prob = root_layer.value(x)
    labels = ([False] * (28 * 28)) + ([True] * 10)
    marginal_prob = root_layer.value(x, labels)
    partition_function = root_layer.value(x, [True] * (28 * 28 + 10))
    normalized_joined_prob = joined_prob - partition_function
    conditional_prob = joined_prob - marginal_prob

    fake_data, sample_probs = root_layer.map_value(tf.zeros([100, 28 * 28 + 10], dtype=tf.float64), [False] * (28 * 28 + 10), True)
    normalized_sample_probs = sample_probs - partition_function
    discriminative_loss = -1 * tf.reduce_sum(conditional_prob, 0)
    generative_loss = -1 * tf.reduce_sum(normalized_joined_prob, 0)
    adversarial_loss = generative_loss + tf.reduce_sum(normalized_sample_probs)
    cost = generative_loss

    train_op = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)
    print('Gradients computed in ' + repr(timer.checkpoint()) + ' seconds')
   # gan_train_data = np.load('mnist-samples.npy')
    with tf.Session() as sess:
        warnings.simplefilter('error', UserWarning)
        saver = tf.train.Saver()
        if restore is not None:
            saver.restore(sess, restore)
        else:
            sess.run(tf.global_variables_initializer())
        if train > 0:
            print('Variables initialized in ' + repr(timer.checkpoint()) + ' seconds')

            # run_metadata = tf.RunMetadata()

            for i in range(train):
                batch = mnist.train.next_batch(100)
                batch = np.concatenate((batch[0], batch[1]), 1)
                # print(partition_function.eval(feed_dict={x: batch}))
                # print(normalized_sample_probs.eval(feed_dict={x: batch}))
                # print(adversarial_loss.eval(feed_dict={x: batch}))
                #start_id = (i * 200) % 6000
                #batch = gan_train_data[start_id:start_id+200, :]
                if i % 50 == 0:
                    training_loss = cost.eval(feed_dict={x: batch})
                    testdata = mnist.test.next_batch(100)
                    # print('partition', partition_function.eval(feed_dict={x: batch[0:1,:]}))
                    # print('joint', joined_prob.eval(feed_dict={x: batch[0:1,:]}))
                    test_loss = cost.eval(feed_dict={x: np.concatenate((testdata[0], testdata[1]), 1)})
                    acc = test_spn(conditional_prob, x, testdata[0], testdata[1])
                    print('Iteration:', i)
                    print('Accuracy:', acc)
                    print('Training loss:', training_loss[0])
                    print('Test loss:', test_loss[0])
                    # print('sample probs', normalized_sample_probs.eval(feed_dict={x: batch})[:10])
                    logfile.write(str(i) + " " + str(acc) + " " + str(training_loss[0])
                                  + " " + str(test_loss[0]) + "\n")
                sess.run([train_op], feed_dict={x: batch})
                #                     options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                #                     run_metadata=run_metadata)
                #        trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                #        trace_file = open('timeline.ctf.json', 'w')
                #        trace_file.write(trace.generate_chrome_trace_format())
            print('Training finished after ' + repr(timer.checkpoint()) + ' seconds')
            testsize = 500
            testdata = mnist.test.next_batch(testsize)
            acc = test_spn(conditional_prob, x, testdata[0], testdata[1])
            print('Accuracy: ', acc)
            if save is not None:
                saver.save(sess, save)
        map_complete_half_image(sess, root_layer, x)
        sample(sess, root_layer, x)


if __name__ == '__main__':
    # train_mnist_model(to_nodes=False, restore='adversarial.ckpt', save='adversarial.ckpt', train=300)
    train_mnist_model(to_nodes=False, adversarial=True, restore=None, save='adversarial.ckpt', train=300, debug=False)
