# -*- coding: utf-8 -*-

import tensorflow as tf

#使用 run方法进行调用
tf.app.flags.DEFINE_string("train_dir", "./model", "训练模型保存位置")
tf.app.flags.DEFINE_string("log_dir", "./logs", "tensofboard 位置")
tf.app.flags.DEFINE_string("ckpt_name", "conversation.ckpt", "模型名字")

tf.app.flags.DEFINE_boolean("train", True, "boolean标志，表示训练或者测试")
tf.app.flags.DEFINE_boolean("test", False, "同上")
tf.app.flags.DEFINE_boolean("data_loop", True, "使用小数据集")
tf.app.flags.DEFINE_integer("batch_size", 100, "batch大小")
tf.app.flags.DEFINE_integer("epoch", 1000, "迭代次数")

tf.app.flags.DEFINE_string("data_path", "./data/chat.log", "训练数据")
tf.app.flags.DEFINE_string("voc_path", "./data/chat.txt", "词汇表")
tf.app.flags.DEFINE_boolean("voc_test", False, "测试词汇表")
tf.app.flags.DEFINE_boolean("voc_build", True, "构建词汇表")

tf.app.flags.DEFINE_integer("max_decode_len", 30, "句子解码长度")


FLAGS = tf.app.flags.FLAGS
