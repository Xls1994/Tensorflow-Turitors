# -*- coding: utf-8 -*-
'''
author:yangyl

'''
# 深度学习入门数据集 Hello World 手写数字识别
# 使用tensorflow的api获取mnist数据

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

# one_hot 形式的标签数据
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)
#########
# 定义占位符
######
# 原始数据大小为28×28的图片，现在转化为一个784的向量
#
X = tf.placeholder(tf.float32, [None, 784])
#  0~9 数字识别属于10分类问题
Y = tf.placeholder(tf.float32, [None, 10])

# 定义隐藏层的神经元个数，也就是矩阵的维度
# 784(输入维度)
#   -> 256 (隐藏层1维度) -> 256 (隐藏层2维度)
#   -> 10 (输出分类的维度)
W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))

# 感知机模型
# W×X，激活函数是relu
L1 = tf.nn.relu(tf.matmul(X, W1))

L2 = tf.nn.relu(tf.matmul(L1, W2))
# 最后输出模型的结果
model = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model,labels=Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)


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
        # 进行mini-batch的梯度下降，更新loss

        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer, feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += sess.run(cost, feed_dict={X: batch_xs, Y: batch_ys})

    print 'Epoch:', '%04d' % (epoch + 1),\
            'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)

print '训练结束!'


#########
# 测试
######
# model输出的结果为一个[10]维的向量.
# tf.argmax 取最大值所在的索引和正确的标签索引对比，计算准确率
# [0.1 0 0 0.7 0 0.2 0 0 0 0] -> 4
def evaluationTok_k(k=3):
    prediction =tf.nn.in_top_k(model,tf.argmax(Y,1),k=k)
    accuracy =tf.reduce_mean(tf.cast(prediction,'float32'))
    is_top1 =tf.equal(tf.nn.top_k(model,k)[1][:,0],tf.cast(tf.argmax(Y,1),'int32'))
    is_top2 = tf.equal(tf.nn.top_k(model, k)[1][:, 1], tf.cast(tf.argmax(Y, 1), 'int32'))
    is_top3 = tf.equal(tf.nn.top_k(model, k)[1][:, 2], tf.cast(tf.argmax(Y, 1), 'int32'))
    is_in_top1 =is_top1
    is_in_top2 = tf.logical_or(is_in_top1, is_top2)
    is_in_top3 = tf.logical_or(is_in_top2, is_top3)
    accuracy11 = tf.reduce_mean(tf.cast(is_in_top1, "float32"))
    accuracy22 = tf.reduce_mean(tf.cast(is_in_top2, "float32"))
    accuracy33 = tf.reduce_mean(tf.cast(is_in_top3, "float32"))
    return  accuracy
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
accuracy_top_k =evaluationTok_k(3)
print '准确率:', sess.run([accuracy,accuracy_top_k],
                            feed_dict={X: mnist.test.images,
Y: mnist.test.labels})
