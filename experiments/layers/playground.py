import tensorflow as tf
import tensorflow.contrib.distributions as dist
import tfspn.layers as layers
import numpy as np
import matplotlib.pyplot as plt

def plot_1d_dist(x, probs, sess, bot=-50, top=150, step=0.25):
    samples = np.arange(bot, top, step, dtype=np.float64)
    samples = np.stack([samples, np.ones_like(samples)], axis=1)
    ys = sess.run(probs, feed_dict={x: samples})
    plt.scatter(samples[:, 0], ys)
    plt.show()


def disc_test():
    disc_gaussians = layers.GaussianLayer([0, 0, 0, 0])
    disc_indicators = layers.BernoulliLayer([1, 1, 1, 1])
    disc_prod_struc = np.zeros((8, 4))
    for i in range(4):
        disc_prod_struc[i, i] = 1
        disc_prod_struc[i + 4, i] = 1
    disc_products = layers.ProductLayer([disc_gaussians, disc_indicators], disc_prod_struc)
    disc_sum = layers.SumLayer([disc_products], np.ones((4, 1), dtype=np.float64))

    x = tf.placeholder(tf.float64, (None, 2))

    testdata = np.zeros((40, 2), dtype=np.float64)
    testdata[:10, 0] = 20
    testdata[10:20, 0] = 70
    testdata[:20, 1] = 1
    testdata[20:30, 0] = 0
    testdata[30:, 0] = 100
    testdata[20:, 1] = 0
    print(testdata)

    disc_joint = disc_sum.value(x)
    disc_marginal = disc_sum.value(x, [False, True])
    disc_conditional = tf.exp(disc_joint - disc_marginal)[:, 0]
    #loss = -tf.reduce_sum(disc_joint - disc_marginal, axis=0)
    loss = -1 * tf.reduce_sum(disc_joint - disc_sum.value(x, [True, True]), 0)
    train_opt = tf.train.AdamOptimizer(learning_rate=0.1).minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            if i % 200 == 0:
                plot_1d_dist(x, disc_conditional, sess)
                print('means', disc_gaussians.means.eval())
                print('stddev', disc_gaussians.stdevs.eval())
                print('bernoulli', disc_indicators.probs.eval())
                print('sum weights', np.transpose(disc_sum.weights.eval()))
                print('loss', loss.eval(feed_dict={x: testdata}))

            sess.run(train_opt, feed_dict={x: testdata})


def sample_test():
    testdata = np.zeros((100, 1), dtype=np.float64)
    testdata[:50, 0] = 20
    testdata[50:, 0] = 70
    testdata = np.random.permutation(testdata)

    leafs = layers.GaussianLayer([0, 0], [0.0, 100.0], [0.5, 0.5])
    sum_layer = layers.SumLayer([leafs], np.array([[1.0], [1.0]]))
    samples, probs = sum_layer.map_value(tf.zeros([100, 1], dtype=tf.float64), [False], sample=True)
    samples = samples[:, 0, :]
    disc_gaussians = layers.GaussianLayer([0, 0, 0, 0], [0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0])
    disc_indicators = layers.DiscreteLayer([1, 1, 1, 1], [0, 1, 0, 1])
    disc_prod_struc = np.zeros((8, 4))
    for i in range(4):
        disc_prod_struc[i, i] = 1
        disc_prod_struc[i + 4, i] = 1
    disc_products = layers.ProductLayer([disc_gaussians, disc_indicators], disc_prod_struc)
    disc_sum = layers.SumLayer([disc_products], np.ones((4, 1), dtype=np.float64))
    print('valid?', disc_products.validate())
    samples_with_fake_labels = tf.concat([samples, tf.zeros_like(samples)], axis=1)
    testdata_with_real_labels = tf.concat([testdata, tf.ones_like(testdata)], axis=1)
    disc_input = tf.concat([samples_with_fake_labels, testdata_with_real_labels], axis=0)
    disc_loss = -1 * disc_sum.value(disc_input)
    gen_loss = -1 * layers.discriminative_cost(samples_with_fake_labels, [False, True], disc_sum)

    # For testing and plotting
    x = tf.placeholder(tf.float64, (None, 2))
    disc_joint = disc_sum.value(x)
    disc_marginal = disc_sum.value(x, [False, True])
    disc_conditional = tf.exp(disc_joint - disc_marginal)[:, 0]



    loss = (30 - samples[0, 0]) * (30 - samples[0, 0])
    train_opt = tf.train.AdamOptimizer(learning_rate=1).minimize(loss)
    #print(sum_layer.var_list())
    gen_training = tf.train.AdamOptimizer(learning_rate=0.1).minimize(gen_loss, var_list=sum_layer.var_list())
    disc_training = tf.train.AdamOptimizer(learning_rate=0.1).minimize(disc_loss, var_list=disc_sum.var_list())
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(10000):
            # sess.run(train_opt)
            sess.run([gen_training, disc_training])
            if i % 100 == 0:
                print('means', leafs.means.eval())
                print('sum weights:', sum_layer.weights.eval())

                print('disc sum weights', disc_sum.weights.eval())
                print('disc means', disc_gaussians.means.eval())
                plot_1d_dist(x, disc_conditional, sess)
                # print(sess.run([samples, probs]))

def relaxed_test():
    var = tf.Variable([-5, -10], dtype=tf.float32)
    cat = dist.RelaxedOneHotCategorical(2.0, logits=var)

    loss = tf.reduce_sum((tf.cast(cat.sample([10]), tf.float32)), axis=0)[0]
    train_op = tf.train.AdamOptimizer().minimize(loss)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            print('Vars:', sess.run(var))
            print('Sample:', sess.run(cat.sample()))
            sess.run(train_op)

disc_test()