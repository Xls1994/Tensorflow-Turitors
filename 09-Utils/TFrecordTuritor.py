# -*- coding: utf-8 -*-
'''
author:yangyl

'''
'''
从TFRecords文件中读取数据， 可以使用tf.TFRecordReader的tf.parse_single_example解析器。
这个操作可以将Example协议内存块(protocol buffer)解析为张量。
'''
import os
import tensorflow as tf
from PIL import Image

cwd =os.getcwd()
classes =['/data/cat','/data/dog']
print cwd
writer =tf.python_io.TFRecordWriter('train.tfre')
for index,name in enumerate(classes):
    class_path =cwd+name+'/'
    for img_name in os.listdir(class_path):
        img_path =class_path +img_name
        img =Image.open(img_path)
        img =img.resize((224,224))
        img_raw =img.tobytes() # convert image to bytes
        feature ={"label":tf.train.Feature(int64_list=tf.train.Int64List(value =[index])),
                   "img_raw": tf.train.Feature(bytes_list =tf.train.BytesList(value=[img_raw]))}
        example =tf.train.Example(features =tf.train.Features(feature=feature))
        writer.write(example.SerializeToString()) #序列化为字符串
writer.close()
import tensorlayer as tl
import numpy as  np
for serialized_example in tf.python_io.tf_record_iterator('train.tfre'):
    example =tf.train.Example()
    example.ParseFromString(serialized_example)
    img_raw =example.features.feature['img_raw'].bytes_list.value
    label =example.features.feature['label'].int64_list.value
    print label
    image = Image.frombytes('RGB', (224, 224), img_raw[0])
    # tl.visualize.frame(np.asarray(image), second=5, saveable=False, name='frame', fig_idx=1283)

def read_and_decode(filename):
    filename_queue =tf.train.string_input_producer([filename])
    reader =tf.TFRecordReader()
    _,serilized_example =reader.read(filename_queue)  #return file name and file
    features = tf.parse_single_example(serialized_example,
                                features={
                                    "label":tf.FixedLenFeature([],tf.int64),
                                    "img_raw":tf.FixedLenFeature([],tf.string)
                                })
    img =tf.decode_raw(features['img_raw'],tf.uint8)
    img =tf.reshape(img,shape=[224,224,3])
    img =tf.cast(img,tf.float32)
    label =tf.cast(features["label"],tf.int32)
    return img,label

img,label =read_and_decode('train.tfre')
img_batch,label_batch =tf.train.shuffle_batch([img,label],batch_size=30,capacity=2000,
                                              min_after_dequeue=1000)
init =tf.global_variables_initializer()
sess =tf.Session()
sess.run(init)
coord =tf.train.Coordinator()
threads =tf.train.start_queue_runners(sess=sess,coord=coord)

for i in range(3):
    val,l=sess.run([img_batch,label_batch])
    print (val.shape,l)

coord.request_stop()
coord.join(threads)
sess.close()





if __name__ == '__main_':
    pass