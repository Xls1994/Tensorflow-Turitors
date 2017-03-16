import tensorflow as tf
import  collections
import math
import os
import random
import zipfile
import numpy as np
import urllib

url ='http://mattmahoney.net/dc/'
def maybe_download(filename,expected_bytes):
    if not os.path.exists(filename):
        filename,_ =urllib.urlretrieve(url+filename,filename)

    stateinfo=os.stat(filename)
    if stateinfo.st_size==expected_bytes:
        print('Found and verified',filename)
    else:
        print stateinfo.st_size
        raise Exception('fail to verfy file')
    return filename
filename =maybe_download('text8.zip',31344016)

def read_data(filename):
    with zipfile.ZipFile(filename)as f:
        data =tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data
words =read_data(filename)
print 'data size',len(words)

vocabulary_size =50000

def build_dataset(words):
    count=[['UNK',-1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary ={}
    for word,_ in count:
        dictionary[word]=len(dictionary)
    data =list()
    unk_count =0
    for word in words:
        if word in dictionary:
            index =dictionary[word]
        else:
            index=0
            unk_count+=1
        data.append(index)
    count[0][1]=unk_count
    reverse_dictionary =dict(zip(dictionary.values(),dictionary.keys()))
    return data,count,dictionary,reverse_dictionary
data,count,dictionary,reverse_dictionary =build_dataset(words)
del words
print 'most common works',count[:5]

data_index =0
def generate_batch(batch_size,num_skips,skip_window):
    global data_index
    assert batch_size%num_skips==0
    assert num_skips<=2*skip_window
    batch =np.ndarray(shape=(batch_size),dtype='int32')
    labels =np.ndarray(shape=(batch_size,1,),dtype='int32')
    span =2*skip_window+1
    buffer =collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    for i in range(batch_size//num_skips):
        target =skip_window
        targets_to_void =[skip_window]
        for j in range(num_skips):
            while target in targets_to_void:
                target =random.randint(0,span-1)
            targets_to_void.append(target)
            batch[i*num_skips+j]=buffer[skip_window]
            labels[i*num_skips+j,0]=buffer[target]
        buffer.append(data[data_index])
        data_index=(data_index+1)%len(data)
    return batch,labels
batch,labels =generate_batch(8,num_skips=2,skip_window=1)
for i in range(8):
    print batch[i],reverse_dictionary[batch[i]],'>',labels[i,0],reverse_dictionary[labels[i,0]]

batch_size =128
embedding_size =128
skip_window =1
num_skips =2

valid_size =16
valid_window =100
valid_examples =np.random.choice(valid_window,valid_size,replace=False)
num_sampled =64

graph =tf.Graph()
with graph.as_default():
    train_inputs =tf.placeholder(tf.int32,shape=[batch_size])
    train_labels =tf.placeholder(tf.int32,shape=[batch_size,1])
    valid_dataset =tf.constant(valid_examples,dtype=tf.int32)
    with tf.device('/cpu:0'):
        embeddings =tf.Variable(tf.random_uniform([vocabulary_size,embedding_size],-1.0,1.0))
        embed =tf.nn.embedding_lookup(embeddings,train_inputs)
        nce_weights =tf.Variable(tf.truncated_normal([vocabulary_size,embedding_size],stddev=1.0/math.sqrt(embedding_size))
                                 )
        nce_biases =tf.Variable(tf.zeros([vocabulary_size]))
        loss =tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights,
                                            biases=nce_biases,
                                            labels=train_labels,
                                            inputs=embed,
                                            num_sampled=num_sampled,
                                            num_classes=vocabulary_size))
        optimizer =tf.train.GradientDescentOptimizer(1.0).minimize(loss)
        norm =tf.sqrt(tf.reduce_sum(tf.square(embeddings),1,keep_dims=True))
        normalized_embeddings =embeddings/norm
        valid_embeddings =tf.nn.embedding_lookup(normalized_embeddings,valid_dataset)
        similarity =tf.matmul(valid_embeddings,normalized_embeddings,transpose_b=True)
        init =tf.global_variables_initializer()
num_steps =1000
with tf.Session(graph=graph)as session:
    init.run()
    print('Initialized')
    averge_loss =0
    for step in range(num_steps):
        batch_inputs,batch_labels =generate_batch(batch_size,num_skips,skip_window)
        feed_dict={train_inputs:batch_inputs,train_labels:batch_labels}
        _,loss_val =session.run([optimizer,loss],feed_dict=feed_dict)
        averge_loss+=loss_val
        if step%2000==0:
            if step>0:
                averge_loss/=2000
            print 'Averge loss at step',step,':',averge_loss
            averge_loss=0