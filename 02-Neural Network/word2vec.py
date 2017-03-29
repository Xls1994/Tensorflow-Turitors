# -*- coding: utf-8 -*-
# Word2Vec 简单的word2vec例子

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


# matplot 一个用用绘图的库


def clean_str(string):
  import re
  """

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
  return string.strip().lower()
# 加载句子
def loadFile(path):
    list =[]
    with open(path,'r')as f:
        for line in f:
            line =line.strip()
            list.append(line)
    return  list
sentences = loadFile('train.sentence')

# 对词进行切分
word_list = " ".join(sentences).split()
word_list = list(set(word_list))
# 定义集合，集合里面包括所有的不重复的词
# 建立词的索引
# 创建一个索引数组，对应索引所在的单词
word_dict = {w: i for i, w in enumerate(word_list)}
word_index = [word_dict[word] for word in word_list]
re_word_index =[w for w, i in word_dict.iteritems()]
# 利用当前词预测上下文的词.
# 今天 天气 很 好 啊
#   -> ([今天, 很], 天气), ([天气, 好], 很), ([很, 啊], 好)
#   -> (天气, 今天), (天气, 很), (很, 天气), (很, 好), (好, 很), (好, 啊)
skip_grams = []

for i in range(1, len(word_index) - 1):
    # (context, target) : ([target index - 1, target index + 1], target)
    target = word_index[i]
    context = [word_index[i - 1], word_index[i + 1]]

    # (target, context[0]), (target, context[1])..
    for w in context:
        skip_grams.append([target, w])


# skip-gram 随机选择一个batch的数据
def random_batch(data, size):
    random_inputs = []
    random_labels = []
    random_index = np.random.choice(range(len(data)), size, replace=False)

    for i in random_index:
        random_inputs.append(data[i][0])  # target
        random_labels.append([data[i][1]])  # context word

    return random_inputs, random_labels


#########
# 参数设置
######
# 迭代次数
training_epoch = 300
# 学习率
learning_rate = 0.1
# batch大小
batch_size = 20
# 词向量大小
embedding_size = 2
# word2vec 利用 nce_loss 优化，随机抽取负例
# batch_size 里取样的个数
num_sampled = 15
# 词典大小
voc_size = len(word_list)


#########
# 初始化
######
inputs = tf.placeholder(tf.int32, shape=[batch_size])
# tf.nn.nce_loss  [batch_size, 1]
labels = tf.placeholder(tf.int32, shape=[batch_size, 1])


# 随机初始化词向量矩阵
#词向量矩阵大小[词典词的个数，维度]
embeddings = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
# 通过lookup进行词向量的映射
#    embeddings     inputs    selected
#    [[1, 2, 3]  -> [2, 3] -> [[2, 3, 4]
#     [2, 3, 4]                [3, 4, 5]]
#     [3, 4, 5]
#     [4, 5, 6]]
selected_embed = tf.nn.embedding_lookup(embeddings, inputs)

# nce_loss 初始化权重和损失
nce_weights = tf.Variable(tf.random_uniform([voc_size, embedding_size], -1.0, 1.0))
nce_biases = tf.Variable(tf.zeros([voc_size]))

# 计算NEC损失函数，每次使用负标签的样本

loss = tf.reduce_mean(
            tf.nn.nce_loss(nce_weights, nce_biases,  labels,selected_embed, num_sampled, voc_size))

train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)


#########
# 训练
######
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)

    for step in range(1, training_epoch + 1):
        batch_inputs, batch_labels = random_batch(skip_grams, batch_size)

        _, loss_val = sess.run([train_op, loss],
                               feed_dict={inputs: batch_inputs,
                                          labels: batch_labels})

        if step % 10 == 0:
            print "loss at step ", step, ": ", loss_val

    # 默认的session里可以利用eval()直接获得张量的值
    #将最后的词向量输出
    trained_embeddings = embeddings.eval()


#########
# 输出Word2Vec，进行可视化
######
    def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
        assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"
        plt.figure(figsize=(18, 18))  # in inches
        for i, label in enumerate(labels):
            x, y = low_dim_embs[i, :]
            plt.scatter(x, y)
            plt.annotate(label,
                         xy=(x, y),
                         xytext=(5, 2),
                         textcoords='offset points',
                         ha='right',
                         va='bottom')

        plt.savefig(filename)

# TSNE 降维操作
    try:
        from sklearn.manifold import TSNE
        import matplotlib.pyplot as plt

        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        plot_only = 500
        low_dim_embs = tsne.fit_transform(trained_embeddings[:plot_only, :])
        labels = [re_word_index[i] for i in xrange(plot_only)]
        plot_with_labels(low_dim_embs, labels)

    except ImportError:
        print("Please install sklearn, matplotlib, and scipy to visualize embeddings.")
# for i, label in enumerate(word_list):
#     x, y = trained_embeddings[i]
#     plt.scatter(x, y)
#     plt.annotate(label, xy=(x, y), xytext=(5, 2),
#                  textcoords='offset points', ha='right', va='bottom')
#
# plt.show()