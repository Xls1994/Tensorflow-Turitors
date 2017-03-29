# -*- coding: utf-8 -*-
# GAN 生成图片
# 构建一个能够生成手写数字的GAN模型


import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 参数设置
######
total_epoch = 1000
batch_size = 100
n_input = 28 * 28
n_noise = 128
n_class = 10


#########
# 占位符定义
######
X = tf.placeholder(tf.float32, [None, n_input])
# 实际图片和噪音数据
Y = tf.placeholder(tf.float32, [None, n_class])
Z = tf.placeholder(tf.float32, [None, n_noise])


def generator(noise, labels, reuse=False):
    with tf.variable_scope('generator'):
        # noise 和 labels 拼接在一起作为G的输入
        inputs = tf.concat([noise, labels],1)

        # 利用Tensorflow的contrib模块实现快速搭建模型
        #然而还是Keras好用
        G1 = tf.contrib.layers.fully_connected(inputs, 256)
        G2 = tf.contrib.layers.fully_connected(G1, n_input)

    return G2


def discriminator(inputs, labels, reuse=None):
    with tf.variable_scope('discriminator') as scope:
        # 在模型中使用和G相同的labels，确保能够判断G生成的图像真假
        # D可以重新使用以前的参数
        if reuse:
            scope.reuse_variables()

        # 同G的输入，拼接inputs和labels
        inputs = tf.concat([inputs, labels],1)

        D1 = tf.contrib.layers.fully_connected(inputs, 256)
        D2 = tf.contrib.layers.fully_connected(D1, 256)
        D3 = tf.contrib.layers.fully_connected(D2, 1, activation_fn=None)

    return D3


# 建立生成器和判别器模型
G = generator(Z, Y)
D_real = discriminator(X, Y)
D_gene = discriminator(G, Y, True)

# 参考下面博客实现交叉熵的GAN模型DCGAN
# http://bamos.github.io/2016/08/09/deep-completion/
# D的损失分为两个部分
loss_D_real = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(D_real),logits=D_real))
loss_D_gene = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(D_gene),logits=D_gene ))
# loss_D_real 和 loss_D_gene的和为D的损失
loss_D = loss_D_real + loss_D_gene
# 同理定义G的损失
loss_G = tf.reduce_mean(
                    tf.nn.sigmoid_cross_entropy_with_logits(labels= tf.ones_like(D_gene),logits=D_gene))

# 将D和G的参数分别收集起来，以便单独训练模型
# 利用get_collection收集在scope范围里的可训练参数
vars_D = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator')
vars_G = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator')

train_D = tf.train.AdamOptimizer().minimize(loss_D, var_list=vars_D)
train_G = tf.train.AdamOptimizer().minimize(loss_G, var_list=vars_G)


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
        noise = np.random.uniform(-1., 1., size=[batch_size, n_noise])

        _, loss_val_D = sess.run([train_D, loss_D], feed_dict={X: batch_xs, Y: batch_ys, Z: noise})
        _, loss_val_G = sess.run([train_G, loss_G], feed_dict={Y: batch_ys, Z: noise})

    print 'Epoch:', '%04d' % (epoch + 1), \
        'D loss: {:.4}'.format(loss_val_D), \
        'G loss: {:.4}'.format(loss_val_G)

    #########
    # 测试
    ######
    if epoch % 10 == 0:
        noise = np.random.uniform(-1., 1., size=[30, n_noise])
        samples = sess.run(G, feed_dict={Y: mnist.validation.labels[:30], Z: noise})

        fig, ax = plt.subplots(6, n_class, figsize=(n_class, 6))

        for i in range(n_class):
            for j in range(6):
                ax[j][i].set_axis_off()

            for j in range(3):
                ax[0+(j*2)][i].imshow(np.reshape(mnist.validation.images[i+(j*n_class)], (28, 28)))
                ax[1+(j*2)][i].imshow(np.reshape(samples[i+(j*n_class)], (28, 28)))

        plt.savefig('samples2/{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        plt.close(fig)


print 'GAN end!'