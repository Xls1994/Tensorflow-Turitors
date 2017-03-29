# -*- coding: utf-8 -*-
# 利用一个2个隐藏层的神经网络进行分类

import tensorflow as tf
# import numpy as np
import pandas as pd
# 导入输入，这里使用的是pandas读取csv文件
# data = np.loadtxt('./data.csv', delimiter=',',
#                   unpack=True, dtype='float32')
data =pd.read_csv('./data.csv',delimiter=',',dtype='float32')
#切分训练数据和标签
x_data =data[['毛皮','翅膀']]
y_data =data.iloc[:,2:]

print y_data

#########
# 定义数据占位符
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 3层神经网络模型，总共6个需要学习的W和b
# 神经元个数
# 2 -> 10 -> 20 -> 3
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
W2 = tf.Variable(tf.random_uniform([10, 20], -1., 1.))
W3 = tf.Variable(tf.random_uniform([20, 3], -1., 1.))

# 利用relu激活神经元
# 第二个隐藏层同样的用法
L1 = tf.nn.relu(tf.matmul(X, W1))
L2 = tf.nn.relu(tf.matmul(L1, W2))

# 模型输出
# 这里使用 softmax_cross_entropy_with_logits 进行激活
# 不需要对输出进行softmax，而且输出不用是one-hot表示
model = tf.matmul(L2, W3)

# 计算损失，使用Adam优化器
cost = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model,name='cost'))

optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


#########
# 初始化会话
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in xrange(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print (step + 1), sess.run(cost, feed_dict={X: x_data, Y: y_data})


#########
# 进行分类预测
# 0: 其他 1: 哺乳动物, 2: 鸟类
######
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print '预测结果:', sess.run(prediction, feed_dict={X: x_data})
print '标签:', sess.run(target, feed_dict={Y: y_data})

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '准确率: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})