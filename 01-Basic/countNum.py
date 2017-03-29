# coding: utf-8
import tensorflow as tf
#创建一个变量，初始化为1
state =tf.Variable(0,name='counter')

one =tf.constant(1)
#定义一个操作，使每次状态+1
new_value =tf.add(state,one)
#assign来进行赋值
update =tf.assign(state,new_value)
#启动会话
sess =tf.Session()
#初始化所有变量
init_op =tf.global_variables_initializer()

sess.run(init_op)
print sess.run(state)

for i in range(10):
    val =sess.run(update)
    print 'i:',val