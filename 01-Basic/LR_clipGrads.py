# -*- coding: utf-8 -*-
# X Y之间的一个回归问题

import tensorflow as tf


x_data = [1, 2, 3]
y_data = [1, 2, 3]

W = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
b = tf.Variable(tf.random_uniform([1], -1.0, 1.0))

X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")

# X Y均为输入数据，利用占位符初始化
# h = W * X + b
# W和 b均是一个变量参数，利用multiply函数代替matmul
#matmul是矩阵的乘法
hypothesis = tf.add(tf.multiply(W, X), b)


# mean(h - Y)^2 : 损失函数，平方误差
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# 利用随机梯度下降优化器进行优化
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
# 最小化损失函数
# train_op = optimizer.minimize(cost)
tvars = tf.trainable_variables()
grads = tf.gradients(cost, tvars)
clip_norm =5
if clip_norm > 0:
    grads, _ = tf.clip_by_global_norm(grads, clip_norm)

train_op = optimizer.apply_gradients(zip(grads, tvars))
# 利用with定义一个默认的sess，不需要显式的关闭这个sess
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # 进行100次迭代
    for step in xrange(100):
        # sess.run 计算损失和进行优化
        # feed_dict：传入训练数据
        _, cost_val = sess.run([train_op, cost], feed_dict={X: x_data, Y: y_data})

        print step, cost_val, sess.run(W), sess.run(b)

    print "\n=== Test ==="
    # 测试拟合的直线效果.
    print "X: 5, Y:", sess.run(hypothesis, feed_dict={X: 5})
    print "X: 2.5, Y:", sess.run(hypothesis, feed_dict={X: 2.5})