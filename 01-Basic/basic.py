# -*- coding: utf-8 -*-
#tensorflow 教程
import tensorflow as tf

 #tf.constant: 定义一个tf的常量
hello = tf.constant('Hello, TensorFlow!')

a = tf.constant(10)
b = tf.constant(32)
c = a + b

# tf.placeholder: 定义占位符，表示待输入的数据
# None 表示这个值可以为任意大小
X = tf.placeholder("float", [None, 3])

# tf.Variable: 定义变量，一般训练过程中的参数使用这个定义
# tf.random_normal: 正态分布
# name: 变量的名字，用于区别不同变量
W = tf.Variable(tf.random_normal([3, 2]), name='Weights')
b = tf.Variable(tf.random_normal([2, 1]), name='Bias')

x_data = [[1, 2, 3], [4, 5, 6]]

# 定义操作
# tf.matmul 矩阵乘法
expr = tf.matmul(X, W) + b

# 启动一个会话，tf的程序默认在sess里运行
sess = tf.Session()
# sess.run: 通过run方法执行特定操作
#  tf.global_variables_initializer 初始化变量
sess.run(tf.global_variables_initializer())

#调用sess.run()，执行不同的操作
print "=== contants ==="
print sess.run(hello)
print "a + b = c =", sess.run(c)
print "=== x_data ==="
print x_data
print "=== W ==="
print sess.run(W)
print "=== b ==="
print sess.run(b)
print "=== expr ==="
# expr
# 输出 expr的结果，占位符需要用字典的方式传入
print sess.run(expr, feed_dict={X: x_data})

# 关闭sess
sess.close()