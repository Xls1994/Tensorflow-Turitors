# coding=utf-8
import numpy as np
#初始化权重
def Xavier(fan_in,fan_out):
    W =np.random.random((fan_in,fan_out))/np.sqrt(fan_in)
    return W

def Xavier2(fan_in,fan_out):
    W =np.random.random((fan_in,fan_out))/np.sqrt(fan_in/2)
    return W

def tfXavier(fan_in, fan_out):
    import tensorflow as tf
    low = -4 * np.sqrt(6.0 / (fan_in + fan_out))  # use 4 for sigmoid, 1 for tanh activation
    high = 4 * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.Variable(tf.random_uniform((fan_in,fan_out), minval=low, maxval=high, dtype=tf.float32))

x =Xavier(3,4)
print x
print x.shape
import tensorflow as tf
Y = tf.placeholder(tf.float32,[2,3])
Z =tf.reshape(Y,[-1])
sess =tf.Session()
sess.run(tf.global_variables_initializer())

y,z =sess.run([Y,Z],feed_dict={Y:np.array([[1,2,3],[4,5,6]])})
print y,z

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.
    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.
    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:
    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var
    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.
    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features
    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':

        # Forward pass
        # Step 1 - shape of mu (D,)
        mu = 1 / float(N) * np.sum(x, axis=0)

        # Step 2 - shape of var (N,D)
        xmu = x - mu

        # Step 3 - shape of carre (N,D)
        carre = xmu**2

        # Step 4 - shape of var (D,)
        var = 1 / float(N) * np.sum(carre, axis=0)

        # Step 5 - Shape sqrtvar (D,)
        sqrtvar = np.sqrt(var + eps)

        # Step 6 - Shape invvar (D,)
        invvar = 1. / sqrtvar

        # Step 7 - Shape va2 (N,D)
        va2 = xmu * invvar

        # Step 8 - Shape va3 (N,D)
        va3 = gamma * va2

        # Step 9 - Shape out (N,D)
        out = va3 + beta

        running_mean = momentum * running_mean + (1.0 - momentum) * mu
        running_var = momentum * running_var + (1.0 - momentum) * var

        cache = (mu, xmu, carre, var, sqrtvar, invvar,
                 va2, va3, gamma, beta, x, bn_param)
    elif mode == 'test':

        mu = running_mean
        var = running_var
        xhat = (x - mu) / np.sqrt(var + eps)
        out = gamma * xhat + beta
        cache = (mu, var, gamma, beta, bn_param)

    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.
    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.
    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.
    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None


    mu, xmu, carre, var, sqrtvar, invvar, va2, va3, gamma, beta, x, bn_param = cache
    eps = bn_param.get('eps', 1e-5)
    N, D = dout.shape

    # Backprop Step 9
    dva3 = dout
    dbeta = np.sum(dout, axis=0)

    # Backprop step 8
    dva2 = gamma * dva3
    dgamma = np.sum(va2 * dva3, axis=0)

    # Backprop step 7
    dxmu = invvar * dva2
    dinvvar = np.sum(xmu * dva2, axis=0)

    # Backprop step 6
    dsqrtvar = -1. / (sqrtvar**2) * dinvvar

    # Backprop step 5
    dvar = 0.5 * (var + eps)**(-0.5) * dsqrtvar

    # Backprop step 4
    dcarre = 1 / float(N) * np.ones((carre.shape)) * dvar

    # Backprop step 3
    dxmu += 2 * xmu * dcarre

    # Backprop step 2
    dx = dxmu
    dmu = - np.sum(dxmu, axis=0)

    # Basckprop step 1
    dx += 1 / float(N) * np.ones((dxmu.shape)) * dmu

    return dx, dgamma, dbeta