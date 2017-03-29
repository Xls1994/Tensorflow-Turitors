# -*- coding: utf-8 -*-
# 利用TensorBorad可视化神经网络

import tensorflow as tf
import numpy as np
import  pandas as pd

data =pd.read_csv('./data.csv',delimiter=',',dtype='float32')
#切分训练数据和标签
x_data =data[['毛皮','翅膀']]
y_data =data.iloc[:,2:]


#########
# 占位符定义
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# with tf.name_scope 定义变量所在的范围
#在这一个作用域下的所有变量都会有一个共同前缀
with tf.name_scope('layer1'):
    W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.), name="W1")
    L1 = tf.nn.relu(tf.matmul(X, W1))

with tf.name_scope('layer2'):
    W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.), name="W2")
    L2 = tf.nn.relu(tf.matmul(L1, W2))

with tf.name_scope('output'):
    W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.), name="W3")
    model = tf.matmul(L2, W3)

with tf.name_scope('optimizer'):
    cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits = model))

    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train_op = optimizer.minimize(cost)

    # tf.summary.scalar 将损失写入到summary中
    tf.summary.scalar('cost', cost)


#########
# 初始化会话
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# 将所有数据汇总到一个操作中
merged = tf.summary.merge_all()
# 写入图表本身和数据具体值的事件文件
writer = tf.summary.FileWriter('./logs', sess.graph)
# 命令行下使用tensorboadr
# tensorboard --logdir=./logs
# 打开网站进行可视化
# http://localhost:6006

for step in xrange(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    # 每次运行merged都会将函数的输出传入事件读写器writer中
    summary = sess.run(merged, feed_dict={X: x_data, Y:y_data})
    writer.add_summary(summary, step)


#########
# 测试
# 0: 其他 1: 哺乳动物 2: 鸟类
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print '预测结果:', sess.run(prediction, feed_dict={X: x_data})
print '标签:', sess.run(target, feed_dict={Y: y_data})

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '准确率: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})