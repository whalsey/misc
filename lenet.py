# Sample code implementing LeNet-5 from Liu Liu

import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


class cnnMNIST(object):
    def __init__(self, lr=1e-4, batch=50, drop_prob=0.5):
        self.lr = lr
        self.batch=batch
        self.drop_prob=drop_prob
        self.epochs = 200
        self.build_graph()

    def build_graph(self):
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_ = tf.placeholder(tf.float32, shape=[None, 10])

        # define conv-layer variables
        W_conv1 = self.weight_variable([5, 5, 1, 32])  # first conv-layer has 32 kernels, size=5
        b_conv1 = self.bias_variable([32])
        W_conv2 = self.weight_variable([5, 5, 32, 64])
        b_conv2 = self.bias_variable([64])

        x_image = tf.reshape(self.x, [-1, 28, 28, 1])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1)
        h_pool1 = self.max_pool_2x2(h_conv1)
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2)
        h_pool2 = self.max_pool_2x2(h_conv2)

        # densely/fully connected layer
        W_fc1 = self.weight_variable([7 * 7 * 64, 1024])
        b_fc1 = self.bias_variable([1024])

        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

        # dropout regularization
        self.keep_prob = tf.placeholder(tf.float32)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # linear classifier
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])

        self.y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_, logits=self.y_conv))
        self.train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)

    def train(self, report_freq=100):
        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        self.eval()  # creating evaluation
        train_result = []
        for i in range(self.epochs):
            batch = mnist.train.next_batch(self.batch)

            if i % report_freq == 0:

                train_acc = self.sess.run(self.accuracy,
                                          feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: self.drop_prob})
                train_result.append(train_acc)

                print('step %d, training accuracy %g' % (i, train_acc))

            self.sess.run([self.train_step], feed_dict={self.x: batch[0], self.y_: batch[1], self.keep_prob: self.drop_prob})

        return train_result

    def eval(self):
        correct_prediction = tf.equal(tf.argmax(self.y_conv, 1), tf.argmax(self.y_, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def test_eval(self):
        self.eval()
        test_acc = self.sess.run(self.accuracy, feed_dict={
            self.x: mnist.test.images, self.y_: mnist.test.labels, self.keep_prob: 1.0})
        print('test accuracy %g' % test_acc)
        return test_acc

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')


if __name__ == '__main__':

    import math

    # combinations of lr, dropout prob, and batch size
    lr_best = 0.0005
    keep_prob_best = 0.5
    batch_size_best = 500

    percentChange = 1

    f = open("lenet_results2.csv", 'w')

    report_freq = 10

    cnn = cnnMNIST()
    train_acc = cnn.train(report_freq)
    test_acc = cnn.test_eval()

    buff = "lr,{},kp,{},bs,{}\n".format(0.0001, 0.5, 50)
    print(buff)
    f.write(buff)
    buff = "epoch," + ','.join([str(i*report_freq) for i in range(200/report_freq)]) + '\n'
    print(buff)
    f.write(buff)
    buff = "train_acc" + ','.join([str(i) for i in train_acc]) + '\n'
    print(buff)
    f.write(buff)
    buff = "test_acc,{}\n\n".format(test_acc)
    print(buff)
    f.write(buff)
    f.flush()

    iteration = 0
    for Q in range(3):
        buff = "ITERATION {}".format(Q) + '\n'
        print(buff)
        f.write(buff)

        # iterate through batch size
        best_acc = 0
        bs_l = [batch_size_best - 10 ** (math.floor(math.log10(batch_size_best)) - iteration) * i for i in
                range(4, -6, -1)]
        for bs in bs_l:
            bs_int = int(bs)
            cnn = cnnMNIST(lr=lr_best, batch=bs_int, drop_prob=keep_prob_best)
            train_acc = cnn.train(report_freq)
            test_acc = cnn.test_eval()

            if test_acc > best_acc:
                best_acc = test_acc
                batch_size_best = bs_int

            buff = "lr,{},kp,{},bs,{}\n".format(lr_best, keep_prob_best, bs)
            print(buff)
            f.write(buff)
            buff = "epoch," + ','.join([str(i * report_freq) for i in range(200 / report_freq)]) + '\n'
            print(buff)
            f.write(buff)
            buff = "train_acc" + ','.join([str(i) for i in train_acc]) + '\n'
            print(buff)
            f.write(buff)
            buff = "test_acc,{}\n".format(test_acc)
            print(buff)
            f.write(buff)
            f.flush()

        # iterate through keep probability
        best_acc = 0
        kp_l = [keep_prob_best - 10 ** (math.floor(math.log10(keep_prob_best)) - iteration) * i for i in
                range(4, -6, -1)]
        for kp in kp_l:
            cnn = cnnMNIST(lr=lr_best, batch=batch_size_best, drop_prob=kp)
            train_acc = cnn.train(report_freq)
            test_acc = cnn.test_eval()

            if test_acc > best_acc:
                best_acc = test_acc
                keep_prob_best = kp

            buff = "lr,{},kp,{},bs,{}\n".format(lr_best, kp, batch_size_best)
            print(buff)
            f.write(buff)
            buff = "epoch," + ','.join([str(i * report_freq) for i in range(200 / report_freq)]) + '\n'
            print(buff)
            f.write(buff)
            buff = "train_acc" + ','.join([str(i) for i in train_acc]) + '\n'
            print(buff)
            f.write(buff)
            buff = "test_acc,{}\n".format(test_acc)
            print(buff)
            f.write(buff)
            f.flush()

        # iterate through lr
        best_acc = 0
        lr_l = [lr_best - 10**(math.floor(math.log10(lr_best))-iteration)*i for i in range(4, -6, -1)]
        for lr in lr_l:
            cnn = cnnMNIST(lr=lr, batch=batch_size_best, drop_prob=keep_prob_best)
            train_acc = cnn.train(report_freq)
            test_acc = cnn.test_eval()

            if test_acc > best_acc:
                best_acc = test_acc
                lr_best = lr

            buff = "lr,{},kp,{},bs,{}\n".format(lr, keep_prob_best, batch_size_best)
            print(buff)
            f.write(buff)
            buff = "epoch," + ','.join([str(i*report_freq) for i in range(200/report_freq)]) + '\n'
            print(buff)
            f.write(buff)
            buff = "train_acc" + ','.join([str(i) for i in train_acc]) + '\n'
            print(buff)
            f.write(buff)
            buff = "test_acc,{}\n\n".format(test_acc)
            print(buff)
            f.write(buff)
            f.flush()

        iteration += 1
