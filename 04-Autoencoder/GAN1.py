# -*- coding: utf-8 -*-
# 2016年无监督学习受到广泛的关注
# Generative Adversarial Network(GAN) 生成式对抗网络简单实现
# https://arxiv.org/abs/1406.2661

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 参数设置
######
total_epoch = 20
batch_size = 100
learning_rate = 0.0002
# 神经网络层参数
n_hidden = 256
n_input = 28 * 28
n_noise = 128  # 噪音数据的大小


#########
# GAN建模
######
# GAN 是 Unsupervised 学习，但不同于 Autoencoder
X = tf.placeholder(tf.float32, [None, n_input])
# GAN使用噪音Z作为输入，企图利用噪音还原真实数据
Z = tf.placeholder(tf.float32, [None, n_noise])

# 生成器G
G_W1 = tf.Variable(tf.random_normal([n_noise, n_hidden], stddev=0.01))
G_b1 = tf.Variable(tf.zeros([n_hidden]))
G_W2 = tf.Variable(tf.random_normal([n_hidden, n_input], stddev=0.01))
G_b2 = tf.Variable(tf.zeros([n_input]))

# 鉴别器D
D_W1 = tf.Variable(tf.random_normal([n_input, n_hidden], stddev=0.01))
D_b1 = tf.Variable(tf.zeros([n_hidden]))
# 鉴别的结果是判断G生成的数据和原始数据的相似程度
D_W2 = tf.Variable(tf.random_normal([n_hidden, 1], stddev=0.01))
D_b2 = tf.Variable(tf.zeros([1]))


# 定义生成器G的函数
def generator(noise_z):
    hidden_layer = tf.nn.relu(tf.matmul(noise_z, G_W1) + G_b1)
    generated_outputs = tf.sigmoid(tf.matmul(hidden_layer, G_W2) + G_b2)
    return generated_outputs


# 定义鉴别器D的函数
def discriminator(inputs):
    hidden_layer = tf.nn.relu(tf.matmul(inputs, D_W1) + D_b1)
    discrimination = tf.sigmoid(tf.matmul(hidden_layer, D_W2) + D_b2)
    return discrimination


# 利用噪音获得G的输入数据
def get_noise(batch_size):
    return np.random.normal(size=(batch_size, n_noise))


# 初始化生成器G和鉴别器D
G = generator(Z)
#对生成器生成的数据进行判别
D_gene = discriminator(G)
# 真实数据产生的鉴别器D_real
D_real = discriminator(X)

# 训练GAN时，要对D和G同时进行训练
# 训练鉴别器D的时候，应该最小化鉴别损失，分为两部分
# 1真实数据鉴别的误差 : tf.log(D_real)
# 2对G生成的数据鉴别的误差 : tf.log(1 - D_gene)
# 这两个误差加在一起，为训练D时的损失函数

loss_D = tf.reduce_mean(tf.log(D_real) + tf.log(1 - D_gene))

# 训练生成器G的时候，应该最大化鉴别误差
#目的是让D判别不出G生成的数据的真假
loss_G = tf.reduce_mean(tf.log(D_gene))

#同时最小化两个损失
D_var_list = [D_W1, D_b1, D_W2, D_b2]
G_var_list = [G_W1, G_b1, G_W2, G_b2]

#利用Adam算法优化参数
train_D = tf.train.AdamOptimizer(learning_rate).minimize(-loss_D, var_list=D_var_list)
train_G = tf.train.AdamOptimizer(learning_rate).minimize(-loss_G, var_list=G_var_list)


#########
# 会话初始化
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

total_batch = int(mnist.train.num_examples/batch_size)
loss_val_D, loss_val_G = 0, 0

for epoch in range(total_epoch):
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        noise = get_noise(batch_size)

        # 进行GAN的训练
        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Z: noise})

    print 'Epoch:', '%04d' % (epoch + 1), \
          'D loss: {:.4}'.format(loss_val_D), \
          'G loss: {:.4}'.format(loss_val_G)

    #########
    # 测试
    ######
    sample_size = 10
    noise = get_noise(sample_size)

    samples = sess.run(G, feed_dict={Z: noise})

    fig, ax = plt.subplots(1, sample_size, figsize=(sample_size, 1))

    for i in range(sample_size):
        ax[i].set_axis_off()
        ax[i].imshow(np.reshape(samples[i], (28, 28)))

    plt.savefig('samples/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
    plt.close(fig)


print 'GAN end!'