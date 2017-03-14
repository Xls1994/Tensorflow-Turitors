import numpy as np
import re
from collections import defaultdict
import pandas as pd
import cPickle
def loadData(path):
    x = cPickle.load(open(path,"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print(len(word_idx_map))
    # print(len(vocab))
    # print (len(revs))
    datasets = make_idx_data_cv(revs, word_idx_map, 1, max_l=101,k=100, filter_h=5)
    img_h = len(datasets[0][0])-1
    test_set_x = datasets[1][:,:img_h]
    test_set_y = np.asarray(datasets[1][:,-1],"int32")
    train_set_x =datasets[0][:,:img_h]
    train_set_y =np.asarray(datasets[0][:,-1],"int32")
    print ('train set size:',np.shape(train_set_x))
    print('embedding size ',np.shape(W))

    return (train_set_x,train_set_y),(test_set_x,test_set_y),W

def batch_iter(data, batch_size, num_epochs):
  """
  Generates a batch iterator for a dataset.
  """
  data = np.array(data)
  data_size = len(data)
  num_batches_per_epoch = int(len(data)/batch_size) + 1
  for epoch in range(num_epochs):
    # Shuffle the data at each epoch
    shuffle_indices = np.random.permutation(np.arange(data_size))
    shuffled_data = data[shuffle_indices]
    for batch_num in range(num_batches_per_epoch):
      start_index = batch_num * batch_size
      end_index = min((batch_num + 1) * batch_size, data_size)
      yield shuffled_data[start_index:end_index]
def make_idx_data_cv(revs, word_idx_map, cv, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, test = [], []
    for rev in revs:
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        if rev["split"]==cv:
            test.append(sent)
        else:
            train.append(sent)
    train = np.array(train,dtype="int")
    test = np.array(test,dtype="int")
    return [train, test]
def build_data_cv(data_folder, clean_string=True):
    """
    Loads data
    """
    revs = []
    train_context_file = data_folder[0]
    train_label_file = data_folder[1]
    test_context_file = data_folder[2]
    test_label_file = data_folder[3]

    trainTag = 0
    testTag = 1

    posTag = "1"
    negPos = "-1"

    vocab = defaultdict(float)
    with open(train_context_file, "r") as f:
        train_label = open(train_label_file, "r")
        for line in f:
            label = train_label.readline().strip();
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            #print(orig_rev)##############
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            polarity = 0
            if label == posTag:
                polarity = 1;
            else:
                polarity = 0;
            datum  = {"y":polarity,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": trainTag}
                      # "split": np.random.randint(0,cv)}
            revs.append(datum)
        train_label.close()
    with open(test_context_file, "r") as f:
        test_label = open(test_label_file, "r")
        for line in f:
            label = test_label.readline().strip();
            rev = []
            rev.append(line.strip())
            if clean_string:
                orig_rev = clean_str(" ".join(rev))
            else:
                orig_rev = " ".join(rev).lower()
            #print(orig_rev)##############
            words = set(orig_rev.split())
            for word in words:
                vocab[word] += 1
            polarity = 0
            if label == posTag:
                polarity = 1;
            else:
                polarity = 0;
            datum  = {"y":polarity,
                      "text": orig_rev,
                      "num_words": len(orig_rev.split()),
                      "split": testTag}
                      # "split": np.random.randint(0,cv)}
            revs.append(datum)
        test_label.close()

    return revs, vocab

def get_W(word_vecs, k=100,path='wordemb'):
    """
    Get word matrix. W[i] is the vector for word indexed by i
    """
    vocab_size = len(word_vecs)
    word_idx_map = dict()
    W = np.zeros(shape=(vocab_size+1, k), dtype='float32')
    W[0] = np.zeros(k, dtype='float32')
    i = 1
    f =open(path+'.word','w')
    for word in word_vecs:
        # print("#############################")
        # print(word_vecs[word].shape)
        # print(word)
        # print(W[i].shape)
        # print("#############################")
        W[i] = word_vecs[word]
        f.write(word+'\n')
        word_idx_map[word] = i
        i += 1
    np.savetxt(path+'.txt',W,fmt='%.7f', delimiter=' ')
    f.close()
    return W, word_idx_map

def load_vec(fname, vocab):
    """
    format: word vec[50]
    """
    word_vecs = {}
    #print(vocab)
    with open(fname, "r") as f:
        for line in f:
            strs =line.strip().split(' ')
            if strs[0] in vocab:

                word_vecs[strs[0]] = np.array([float(elem) for elem in strs[1:]], dtype='float32')

    return word_vecs

def add_unknown_words(word_vecs, vocab, min_df=1, k=100):
    """
    For words that occur in at least min_df documents, create a separate word vector.
    0.25 is chosen so the unknown vectors have (approximately) same variance as pre-trained ones
    """
    for word in vocab:
        if word not in word_vecs and vocab[word] >= min_df:
            # print("************************")
            # print(word)
            # print("************************")
            word_vecs[word] = np.random.uniform(-0.25,0.25,k)

def clean_str(string, TREC=False):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip() if TREC else string.strip().lower()

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x
def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes) to binary class matrix, for use with categorical_crossentropy.

    # Arguments
        y: class vector to be converted into a matrix
        nb_classes: total number of classes

    # Returns
        A binary matrix representation of the input.
    '''
    y = np.array(y, dtype='int')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i, y[i]] = 1.
    return Y