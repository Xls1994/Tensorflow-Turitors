# -*- coding: utf-8 -*-
#  定义一个简单的三值分类问题

import tensorflow as tf
import numpy as np

# 训练数据初始化
#[毛 翅膀]
x_data = np.array(
    [[0, 0], [1, 0], [1, 1], [0, 0], [0, 0], [0, 1]])

# [其他, 哺乳动物, 鸟类]
# 标签为one-hot形式的
y_data = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0],
    [1, 0, 0],
    [0, 0, 1]
])

#########
# 定义数据占位符
######
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# 定义变量 [输入维度, 隐藏层维度] -> [2, 10] .
W1 = tf.Variable(tf.random_uniform([2, 10], -1., 1.))
# 第二层变量 [隐藏层维度, 输入维度] -> [10, 3]
W2 = tf.Variable(tf.random_uniform([10, 3], -1., 1.))

# 偏置初始化，这里仅仅初始化为0

b1 = tf.Variable(tf.zeros([10]))
b2 = tf.Variable(tf.zeros([3]))

# 逻辑回归，矩阵乘法
L = tf.add(tf.matmul(X, W1), b1)

# 使用ReLu的激活函数进行激活
# relu： max（0,x）
L = tf.nn.relu(L)

# 최第二层输出层
# add 函数会把b加到L ×W2上
model = tf.add(tf.matmul(L, W2), b2)
# 利用softmax函数进行概率求解
# 每个维度的概率之和为1,最大的概率为输出的类型
# [8.04, 2.76, -6.52] -> [0.53 0.24 0.23]
model = tf.nn.softmax(model)

# 利用交叉熵计算损失，然后使用随机梯度下降进行优化
# 使用reduce_mean获得平均值作为损失
#reduce_sum 进行求和操作 axis为进行求和的轴 0为列，1为行即每行相加
#        Y         model         Y * tf.log(model)   reduce_sum(axis=1)
#     [[1 0 0]  [[0.1 0.7 0.2]  -> [[-1.0  0    0]  -> [-1.0, -0.09]
#     [0 1 0]]  [0.2 0.8 0.0]]     [ 0   -0.09 0]]

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(model), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)


#########
# 启动会话，训练模型
######
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for step in xrange(100):
    sess.run(train_op, feed_dict={X: x_data, Y: y_data})

    if (step + 1) % 10 == 0:
        print (step + 1), sess.run(cost, feed_dict={X: x_data, Y: y_data})


#########
# 进行测试
# 0: 其他 1: 哺乳动物, 2: 鸟
######
# tf.argmax: 选择值最大的索引，并返回
# 例如) [[0 1 0] [1 0 0]] -> [2 1]
#      [[0.2 0.7 0.1] [0.9 0.1 0.]] -> [2 1]
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
print '预测结果:', sess.run(prediction, feed_dict={X: x_data})
print '实际结果:', sess.run(target, feed_dict={Y: y_data})

check_prediction = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '准确率: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data})