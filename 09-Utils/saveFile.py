# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

# save and restore variable from the files
# 需要保存在一共文件夹里面 否则读取失败
def saveFile(path='tmp/save_net.ckpt'):
    W =tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weights')
    b =tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')

    init =tf.initialize_all_variables()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        savepath =saver.save(sess,path)
        print 'Save to'
def loadFile(path):
    W =tf.Variable(np.arange(6).reshape((2,3)),dtype =tf.float32,name='weights')
    b = tf.Variable (np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')
    saver =tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess,path)
        print 'Weights',np.shape(sess.run(W))
        print 'biases', sess.run(b)


if __name__=="__main__":
    path ='tmp/save_net.ckpt'
    loadFile(path)
