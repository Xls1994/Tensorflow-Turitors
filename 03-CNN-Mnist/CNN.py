# -*- coding: utf-8 -*-
# 利用卷积神经网络进行手写数字识别

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 占位符初始化
######
# 数据集是28×28的灰度图片，在tf里的格式为[sample,width,height,channel]
# CNN接收输入的是三维矩阵
X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])

# 初始化卷积核
# W1 [3 3 1 32] -> [3 3]: 卷积核的大小, 1: X的通道数channel, 32: 卷积核的个数
# L1 Conv shape=(?, 28, 28, 32)
#    Pool     ->(?, 14, 14, 32)
W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))
# tf.nn.conv2d 实现卷积操作
# padding='SAME' 输出图像大小和原来相同
L1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides=[1, 1, 1, 1], padding='SAME'))
# Pooling 进行maxpooling操作
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, 0.8)

# L2 Conv shape=(?, 14, 14, 64)
#    Pool     ->(?, 7, 7, 64)
#    Reshape  ->(?, 256)
# W2  [3, 3, 32, 64]  32 为 L1 的卷积核 W1 的数量, 64是W2的数量
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
L2 = tf.nn.relu(tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME'))
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
# Full Connect 将池化后的输出进行reshape，形成全连接层
# Pooling (?, 7, 7, 64) .
L2 = tf.reshape(L2, [-1, 7 * 7 * 64])
L2 = tf.nn.dropout(L2, 0.8)

# FC 全连接层 7x7x64 ->  256
W3 = tf.Variable(tf.random_normal([7 * 7 * 64, 256], stddev=0.01))
L3 = tf.nn.relu(tf.matmul(L2, W3))
L3 = tf.nn.dropout(L3, 0.5)

# 将 L3 全连接层 256个神经元用于0~9的分类，输出为10个神经元
W4 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))
model = tf.matmul(L3, W4)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model,name='cost'))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

# optimizer = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cost)


#########
# 会话初始化
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

batch_size = 100
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(15):
    total_cost = 0

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 이미지 데이터를 CNN 모델을 위한 자료형태인 [28 28 1] 의 형태로 재구성합니다.
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)

print '训练完毕!'


#########
# 결과 확인
######
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '准确率:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
Y: mnist.test.labels})