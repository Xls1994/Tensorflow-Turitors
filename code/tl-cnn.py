import tensorflow as tf
import tensorlayer as tl
import  numpy as np
sess = tf.InteractiveSession()


# X_train, y_train, X_val, y_val, X_test, y_test = \
#                                 tl.files.load_mnist_dataset(shape=(-1,784))
X_train =np.asarray([[1,2,3,4],[1,2,2,3],[2,3,4,1]],dtype='int32')
y_train =np.asarray([1,2,0],dtype='int32')

x = tf.placeholder(tf.int32, shape=[None, 4], name='x')
y_ = tf.placeholder(tf.int64, shape=[None,], name='y_')

# network = tl.layers.InputLayer(x, name='input_layer')
network =tl.layers.EmbeddingInputlayer(x,vocabulary_size=5,embedding_size=10)
network = tl.layers.DropoutLayer(network, keep=0.8, name='drop1')
# print tf.shape(network.outputs)
network =tl.layers.ReshapeLayer(network,shape=[-1,4,10,1],name='reshape_layer')
network =tl.layers.Conv2dLayer(network,shape=[3,10,1,100],strides=[1,1,1,1],padding='SAME',name='cnn-layer1')

network =tl.layers.PoolLayer(network,strides=[1,2,2,1])
network =tl.layers.FlattenLayer(network)
network = tl.layers.DenseLayer(network, n_units=100,
                                act = tf.nn.relu, name='relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop2')
network = tl.layers.DenseLayer(network, n_units=50,
                                act = tf.nn.relu, name='relu2')
network =tl.layers.FlattenLayer(network,name='flatten2')
network = tl.layers.DropoutLayer(network, keep=0.5, name='drop3')
network = tl.layers.DenseLayer(network, n_units=3,
                                act = tf.nn.softmax,
                                name='output_layer')

y = network.outputs

cost = tl.cost.cross_entropy(y, y_)
correct_prediction = tf.equal(tf.argmax(y, 1), y_)
acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_op = tf.argmax(y, 1)


train_params = network.all_params
train_op = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999,
                            epsilon=1e-08, use_locking=False).minimize(cost, var_list=train_params)


sess.run(tf.global_variables_initializer())
# sess.run(tf.initialize_all_variables())

network.print_params()
network.print_layers()
feed_dict ={x:X_train,y_:y_train}
feed_dict.update( network.all_drop )
_,cost =sess.run([train_op,cost],feed_dict=feed_dict)
print cost
# tl.utils.fit(sess, network, train_op, cost, X_train, y_train, x, y_,
#             acc=acc, batch_size=1, n_epoch=5, print_freq=1,
#              eval_train=False)


# tl.utils.test(sess, network, acc, X_test, y_test, x, y_, batch_size=None, cost=cost)


tl.files.save_npz(network.all_params , name='model.npz')
sess.close()