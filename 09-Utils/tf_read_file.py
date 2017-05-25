# -*- coding: utf-8 -*-
'''
author:yangyl

'''
import tensorflow as tf
import os

writer =tf.python_io.TFRecordWriter("train.record")
fileName ='./data/test.csv'
with open(fileName,'r')as f:
    for line in f:
        array =line.split(",")
        data =[int (x)for x in array[:-2]]
        label =[int(x)for x in array[-2:]]

        example = tf.train.Example(features=tf.train.Features(feature={
            "label": tf.train.Feature(int64_list=tf.train.Int64List(value=label)),
            'data': tf.train.Feature(int64_list=tf.train.Int64List(value=data))
        }))
        print example
        writer.write(example.SerializeToString())
writer.close()

for serialized_example in tf.python_io.tf_record_iterator("train.record"):
    example = tf.train.Example()
    example.ParseFromString(serialized_example)

    data = example.features.feature['data'].int64_list.value
    label = example.features.feature['label'].int64_list.value
    # 可以做一些预处理之类的
    print "data:",data, "label: ",label
if __name__ == '__main__':
    pass