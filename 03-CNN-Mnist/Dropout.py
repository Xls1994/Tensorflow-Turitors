# -*- coding: utf-8 -*-
# dropout 小技巧
#没什么特别之处，主要是dropout的应用。注意tf里面的数字是保留的概率
#真在drop的概率为1-keep_prob

import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 占位符初始化
######
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

W1 = tf.Variable(tf.random_normal([784, 256], stddev=0.01))
W2 = tf.Variable(tf.random_normal([256, 256], stddev=0.01))
W3 = tf.Variable(tf.random_normal([256, 10], stddev=0.01))

L1 = tf.nn.relu(tf.matmul(X, W1))
# 使用dropout
# 注意dropout的参数！
L1 = tf.nn.dropout(L1, 0.8)
L2 = tf.nn.relu(tf.matmul(L1, W2))
L2 = tf.nn.dropout(L2, 0.8)
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
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '准确率:', sess.run(accuracy,
                            feed_dict={X: mnist.test.images,
Y: mnist.test.labels})