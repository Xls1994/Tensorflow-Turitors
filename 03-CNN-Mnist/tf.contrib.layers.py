# -*- coding: utf-8 -*-
#tf.contrib.layers封装了一些自己定义好的层，可以直接使用
#这是一个利用layers进行CNN训练的例子
#我们推荐使用Keras进行tensorflow的二次封装，Keras的详细例子请查看：https://keras.io/
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./mnist/data/", one_hot=True)


#########
# 参数初始化
######
n_width = 28  # MNIST 图像宽度
n_height = 28  # MNIST 图像高度
n_output = 10

#########
# 占位符初始化
######
X = tf.placeholder(tf.float32, [None, n_width, n_height, 1])
Y = tf.placeholder(tf.float32, [None, n_output])

# 定义 inputs, outputs size, kernel_size 就可以使用

L1 = tf.contrib.layers.conv2d(X, 32, [3, 3])
L2 = tf.contrib.layers.max_pool2d(L1, [2, 2])
# normalizer_fn 进行正规化的函数，例如dropout
L3 = tf.contrib.layers.conv2d(L2, 64, [3, 3],
                              normalizer_fn=tf.nn.dropout,
                              normalizer_params={'keep_prob': 0.8})
L4 = tf.contrib.layers.max_pool2d(L3, [2, 2])

L5 = tf.contrib.layers.flatten(L4)
L5 = tf.contrib.layers.fully_connected(L5, 256,
                                       normalizer_fn=tf.contrib.layers.batch_norm)
model = tf.contrib.layers.fully_connected(L5, n_output)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y,logits=model,name='cost'))
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
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        # 输入数据reshape
        batch_xs = batch_xs.reshape(-1, 28, 28, 1)
        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y: batch_ys})
        total_cost += cost_val

    print 'Epoch:', '%04d' % (epoch + 1), \
        'Avg. cost =', '{:.3f}'.format(total_cost / total_batch)

print '训练结束!'


#########
# 测试
######
check_prediction = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(check_prediction, tf.float32))
print '准确率:', sess.run(accuracy,
                       feed_dict={X: mnist.test.images.reshape(-1, 28, 28, 1),
Y: mnist.test.labels})