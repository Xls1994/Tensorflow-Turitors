# -*- coding: utf-8 -*-
# 动态RNN Dynamic RNN 用于句子生成
#可以不用指定每个时间步的长度

import tensorflow as tf
import numpy as np


num_arr = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0']
num_dic = {n: i for i, n in enumerate(num_arr)}
dic_len = len(num_dic)

# 尝试预测不同的序列
# 123 -> X, 4 -> Y
# 12 -> X, 3 -> Y
seq_data = ['1234', '2345', '3456', '4567', '5678', '6789', '7890']
seq_data2 = ['123', '234', '345', '456', '567', '678', '789', '890']


def one_hot_seq(seq_data):
    x_batch = []
    y_batch = []
    for seq in seq_data:
        x_data = [num_dic[n] for n in seq[:-1]]
        y_data = num_dic[seq[-1]]
        x_batch.append(np.eye(dic_len)[x_data])
        # 标签如果不是one-hot形式 需要使用特殊的损失函数
        # sparse_softmax_cross_entropy_with_logits
        y_batch.append([y_data])

    return x_batch, y_batch


#########
# 参数设置
######
n_input = 10
n_classes = 10
n_hidden = 128
# RNN 层数设置.
n_layers = 3


#########
# 占位符初始化
######
# Dynamic RNN 允许时间步为None
# [batch size, time steps, input size]
X = tf.placeholder(tf.float32, [None, None, n_input])
# 使用sparse_softmax_cross_entropy_with_logits 计算损失
# 输出的值如下
# [batch size, time steps]
Y = tf.placeholder(tf.int32, [None, 1])

W = tf.Variable(tf.random_normal([n_hidden, n_classes]))
b = tf.Variable(tf.random_normal([n_classes]))

# tf.nn.dynamic_rnn 的 time_major 应该设置为True
# 对输入Tensor进行转置
# Tensor[batch_size, n_steps, n_input] -> Tensor[n_steps, batch_size, n_input]
X_t = tf.transpose(X, [1, 0, 2])

# RNN cell.
# 进行Dropout
cell = tf.contrib.rnn.BasicRNNCell(n_hidden)
cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.5)
# 设置多层RNN 结构 包装多个cell
cell = tf.contrib.rnn.MultiRNNCell([cell] * n_layers)

# time_major ==Flase:[batch_size,max_time,depth]
#time_major ==True: [n_steps,batch_size,depth]
outputs, states = tf.nn.dynamic_rnn(cell, X_t, dtype=tf.float32, time_major=True)
#output:time_major ==False [batch_size,n_steps,cell.output_size]
#output:time_jor ==True [n_steps,batch_size,cell.output_size ]
# logits
logits = tf.matmul(outputs[-1], W) + b

# 标签不是one-hot形式 需要reshape Tensor [batch_size*1]
# logits 的shape为[batch_size, n_classes]
labels = tf.reshape(Y, [-1])


cost = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))

train_op = tf.train.RMSPropOptimizer(learning_rate=0.01).minimize(cost)


#########
# 会话初始化
######
sess = tf.Session()
sess.run(tf.global_variables_initializer())

x_batch, y_batch = one_hot_seq(seq_data)
x_batch2, y_batch2 = one_hot_seq(seq_data2)

for epoch in range(30):
    _, loss4 = sess.run([train_op, cost], feed_dict={X: x_batch, Y: y_batch})
    _, loss3 = sess.run([train_op, cost], feed_dict={X: x_batch2, Y: y_batch2})

    print 'Epoch:', '%04d' % (epoch + 1), 'cost =', \
        'bucket[4] =', '{:.6f}'.format(loss4), \
        'bucket[3] =', '{:.6f}'.format(loss3)

print '训练完毕!'


#########
# 测试
######
# 定义测试阶段的预测函数
def prediction(seq_data):
    prediction = tf.cast(tf.argmax(logits, 1), tf.int32)
    prediction_check = tf.equal(prediction, labels)
    accuracy = tf.reduce_mean(tf.cast(prediction_check, tf.float32))

    x_batch_t, y_batch_t = one_hot_seq(seq_data)
    real, predict, accuracy_val = sess.run([labels, prediction, accuracy],
                                           feed_dict={X: x_batch_t, Y: y_batch_t})

    print "\n=== 测试结果 ==="
    print '原始数据:', seq_data
    print '标签:', [num_arr[i] for i in real]
    print '预测结果:', [num_arr[i] for i in predict]
    print '准确率:', accuracy_val


# 输入数据进行预测
seq_data_test = ['123', '345', '789']
prediction(seq_data_test)

seq_data_test = ['1234', '2345', '7890']
prediction(seq_data_test)

# 输入数据2
seq_data_test = ['23', '78', '90']
prediction(seq_data_test)

seq_data_test = ['12345', '34567', '67890']
prediction(seq_data_test)