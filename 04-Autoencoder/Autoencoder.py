# -*- coding: utf-8 -*-
# 无监督学习的自动编码机模型(Unsupervised)  Autoencoder

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 参数设置
######
learning_rate = 0.01
training_epoch = 20
batch_size = 100
# 隐藏神经元个数
n_hidden = 256  # 隐藏层
n_input = 28*28   # 输入图像的大小，利用输入图像进行重构


#########
# 占位符初始化
######
# Y是不存在的，因为输入和输出是一样的
X = tf.placeholder(tf.float32, [None, n_input])

# 设置编码和解码层的权重和偏置
# 整个模型的过程：输入-编码-解码-输出
# input -> encode -> decode -> output
W_encode = tf.Variable(tf.random_normal([n_input, n_hidden]))
b_encode = tf.Variable(tf.random_normal([n_hidden]))
# encode 将输入的数据编码为一个高维的特征数据
# decode 将这个高维的特征数据解码为和输入一样的数据，实现重构
# 可以通过自定义隐藏层结构和输入输出来进行自动编码机的建模
W_decode = tf.Variable(tf.random_normal([n_hidden, n_input]))
b_decode = tf.Variable(tf.random_normal([n_input]))

# sigmoid 进行数据激活
# sigmoid(X * W + b)
# 模型建模过程
encoder = tf.nn.sigmoid(
                tf.add(tf.matmul(X, W_encode), b_encode))
# 解码过程和编码相反

decoder = tf.nn.sigmoid(
                tf.add(tf.matmul(encoder, W_decode), b_decode))

#自动编码机就是要利用输入来进行重构，输入数据和解码数据的差异就是编码机的损失
#这里使用的是平方差损失‘mse’
Y = X

cost = tf.reduce_mean(tf.pow(Y - decoder, 2))
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)


#########
# 会话初始化
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(training_epoch):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs})
        total_cost += cost_val

    print 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.6f}'.format(total_cost / total_batch)

print '训练结束'


#########
# 测试
# 查看自动编码机重构后的图像
######
sample_size = 10

samples = sess.run(decoder, feed_dict={X: mnist.test.images[:sample_size]})

fig, ax = plt.subplots(2, sample_size, figsize=(sample_size, 2))

for i in range(sample_size):
    ax[0][i].set_axis_off()
    ax[1][i].set_axis_off()
    ax[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))
    ax[1][i].imshow(np.reshape(samples[i], (28, 28)))

plt.show()