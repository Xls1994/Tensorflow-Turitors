# -*- coding: utf-8 -*-

import tensorflow as tf
import random
import math
import os

from config import FLAGS
from model import Seq2Seq
from dialog import Dialog


def train(dialog, batch_size=100, epoch=100):
    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        # TODO: 加载一个会话  可以利用summary 恢复模型数据
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print "模型检查点位置..", ckpt.model_checkpoint_path
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print "初始化会话"
            sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(FLAGS.log_dir, sess.graph)

        total_batch = int(math.ceil(len(dialog.examples)/float(batch_size)))

        for step in range(total_batch * epoch):
            enc_input, dec_input, targets = dialog.next_batch(batch_size)

            _, loss = model.train(sess, enc_input, dec_input, targets)

            if (step + 1) % 100 == 0:
                model.write_logs(sess, writer, enc_input, dec_input, targets)

                print 'Step:', '%06d' % model.global_step.eval(),\
                      'cost =', '{:.6f}'.format(loss)
        # saver 用于保存和加载数据
        checkpoint_path = os.path.join(FLAGS.train_dir, FLAGS.ckpt_name)
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)

    print '训练完成!'


def test(dialog, batch_size=100):
    print "\n=== 测试 ==="

    model = Seq2Seq(dialog.vocab_size)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
        print "模型检查点位置.", ckpt.model_checkpoint_path
        model.saver.restore(sess, ckpt.model_checkpoint_path)

        enc_input, dec_input, targets = dialog.next_batch(batch_size)

        expect, outputs, accuracy = model.test(sess, enc_input, dec_input, targets)

        expect = dialog.decode(expect)
        outputs = dialog.decode(outputs)

        pick = random.randrange(0, len(expect) / 2)
        input = dialog.decode([dialog.examples[pick * 2]], True)
        expect = dialog.decode([dialog.examples[pick * 2 + 1]], True)
        outputs = dialog.cut_eos(outputs[pick])

        print "\n准确率:", accuracy
        print "数据展示\n",
        print "    输入数据:", input
        print "    答案:", expect
        print "    实际输出:", ' '.join(outputs)


def main(_):
    dialog = Dialog()

    dialog.load_vocab(FLAGS.voc_path)
    print dialog.vocab_list
    dialog.load_examples(FLAGS.data_path)

    if FLAGS.train:
        train(dialog, batch_size=FLAGS.batch_size, epoch=FLAGS.epoch)
    elif FLAGS.test:
        test(dialog, batch_size=FLAGS.batch_size)

if __name__ == "__main__":
    tf.app.run()
