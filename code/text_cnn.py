import tensorflow as tf

# highway layer that borrowed from https://github.com/carpedm20/lstm-char-cnn-tensorflow
def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
  """Highway Network (cf. http://arxiv.org/abs/1505.00387).

  t = sigmoid(Wy + b)
  z = t * g(Wy + b) + (1 - t) * y
  where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
  """
  output = input_
  for idx in xrange(layer_size):
    output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))

    transform_gate = tf.sigmoid(
      tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)
    carry_gate = 1. - transform_gate

    output = transform_gate * output + carry_gate * input_

  return output


class TextCNN(object):
  """
  A CNN for text classification.
  Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
  """
  def __init__(
    self, sequence_length, num_classes, vocab_size,
    embedding_size, filter_sizes, num_filters, embeddingVec,l2_reg_lambda=0.0):
      # Placeholders for input, output and dropout
      self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
      self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
      self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

      # Keeping track of l2 regularization loss (optional)
      l2_loss = tf.constant(0.0)

      # Embedding layer
      with tf.device('/cpu:0'), tf.name_scope("embedding"):
        W = tf.Variable(initial_value=embeddingVec,name="W")
        self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
        self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

      # Create a convolution + maxpool layer for each filter size
      pooled_outputs = []
      for filter_size, num_filter in zip(filter_sizes, num_filters):
        with tf.name_scope("conv-maxpool-%s" % filter_size):
          # Convolution Layer
          filter_shape = [filter_size, embedding_size, 1, num_filter]
          W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
          b = tf.Variable(tf.constant(0.1, shape=[num_filter]), name="b")
          conv = tf.nn.conv2d(
            self.embedded_chars_expanded,
            W,
            strides=[1, 1, 1, 1],
            padding="VALID",
            name="conv")
          # Apply nonlinearity
          h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
          # Maxpooling over the outputs
          pooled = tf.nn.max_pool(
            h,
            ksize=[1, sequence_length - filter_size + 1, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")
          pooled_outputs.append(pooled)

      # Combine all the pooled features
      num_filters_total = sum(num_filters)
      self.h_pool = tf.concat(3, pooled_outputs)
      self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

      # Add highway
      with tf.name_scope("highway"):
        self.h_highway = highway(self.h_pool_flat, self.h_pool_flat.get_shape()[1], 1, 0)

      # Add dropout
      with tf.name_scope("dropout"):
        self.h_drop = tf.nn.dropout(self.h_highway, self.dropout_keep_prob)

      # Final (unnormalized) scores and predictions
      with tf.name_scope("output"):
        W = tf.Variable(tf.truncated_normal([num_filters_total, num_classes], stddev=0.1), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
        l2_loss += tf.nn.l2_loss(W)
        l2_loss += tf.nn.l2_loss(b)
        outputs =tf.matmul(self.h_drop,W)+b
        self.scores = tf.nn.softmax(outputs)
        # self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
        self.predictions = tf.argmax(self.scores, 1, name="predictions")

      # CalculateMean cross-entropy loss
      with tf.name_scope("loss"):
        losses = tf.nn.softmax_cross_entropy_with_logits(self.scores, self.input_y)
        self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

      # Accuracy
      with tf.name_scope("accuracy"):
        correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")
