# -*- coding: utf-8 -*-
# 使用RNN来进行序列预测，理解在自然语言处理领域（NLP）的简单应用
# 预测0-10,构建一个RNN模型

import tensorflow as tf
import numpy as np


num_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
# 建立字典，然后构建one-hot形式的数据
# {'1': 0, '2': 1, '3': 2, ..., '9': 9, '0', 10}
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# 构建输入输出的数据，X为输入。Y为输出
# 123 -> X, 4 -> Y
# 234 -> X, 5 -> Y
seq_data = ['1234', '2345', '3456', '4567', '5678', '6789', '7890']


# 将数据转化为one-hot形式的编码
def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    for seq in seq_data:
        # x_data 和y_data的形式分别为：
        # [0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5] ...
        x_data = [num_dic[n] for n in seq[:-1]]
        # 3, 4, 5, 6...10
        y_data = num_dic[seq[-1]]
        # one-hot coding
        # if x_data is [0, 1, 2]:
        # [[ 1.  0.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  1.  0.  0.  0.  0.  0.  0.  0.  0.]
        #  [ 0.  0.  1.  0.  0.  0.  0.  0.  0.  0.]]
        x_batch.append(np.eye(dic_len)[x_data])
        # if 3: [ 0.  0.  0.  1.  0.  0.  0.  0.  0.  0.]
        y_batch.append(np.eye(dic_len)[y_data])

    return x_batch, y_batch


#########
# 参数设置
######
# 输入大小，将一个数映射成10维的向量
#  3 => [0 0 1 0 0 0 0 0 0 0 0]
n_input = 10
# 模型学习的目标: [1 2 3] => 3
# RNN 时间步的步长.
n_steps = 3
# 输出的类别
n_classes = 10
# 隐藏层单元
n_hidden = 128


#########
# 占位符初始化
######
X = tf.placeholder(tf.float32, [None, n_steps, n_input])
Y = tf.placeholder(tf.float32, [None, n_classes])

W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# RNN 输入数据为三维
# [batch_size, n_steps, n_input]
#    -> Tensor[n_steps, batch_size, n_input]
X_t = tf.transpose(X, [1, 0, 2])
# #    -> Tensor[n_steps*batch_size, n_input]
X_t = tf.reshape(X_t, [-1, n_input])
# #    -> [n_steps, Tensor[batch_size, n_input]]
#
X_t =tf.split(X_t,n_steps,0)
# RNN 初始化
# print tf.shape(X_t)
# BasicRNNCell,BasicLSTMCell,GRUCell

cell = tf.contrib.rnn.BasicLSTMCell(n_hidden,forget_bias=1.0)
outputs,states =tf.contrib.rnn.static_rnn(cell,X_t,dtype=tf.float32)




#  输出
logits = tf.matmul(outputs[-1], W) + b

cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=logits))

train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)


#########
# 会话初始化
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch = one_hot_seq(seq_data)

for epoch in range(100):
    _, loss = sess.run([train_op, cost], feed_dict={X: x_batch, Y: y_batch})

    # 输出训练结果
    print sess.run(tf.argmax(logits, 1), feed_dict={X: x_batch, Y: y_batch})
    print sess.run(tf.argmax(Y, 1), feed_dict={X: x_batch, Y: y_batch})

    print 'Epoch:', '%04d' % (epoch + 1), \
        'cost =', '{:.6f}'.format(loss)

print '训练结束!'


#########
# 测试
######
prediction = tf.argmax(logits, 1)
prediction_check = tf.equal(prediction, tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

seq_data = ['1234', '3456', '6789', '7890']
x_batch, y_batch = one_hot_seq(seq_data)

real, predict, accuracy_val = sess.run([tf.argmax(Y, 1), prediction, accuracy],
                                       feed_dict={X: x_batch, Y: y_batch})

print "\n=== 测试 ==="
print '原始数据:', seq_data
print '标签:', [num_arr[i] for i in real]
print '预测:', [num_arr[i] for i in predict]
print '准确率:', accuracy_val