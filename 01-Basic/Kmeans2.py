import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
num_puntos = 2000
conjunto_puntos = []
for i in range(num_puntos):
    if np.random.random() > 0.5:
        x, y = np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)
        conjunto_puntos.append([x, y])
    else:
        x, y = np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)
        conjunto_puntos.append([x, y])
# df = pd.DataFrame({'x': [v[0] for v in conjunto_puntos], 'y':
#     [v[1] for v in conjunto_puntos]})
# sns.lmplot('x', 'y', data=df, fit_reg=False, size=6)
# plt.show()

import time

N=2000
K=4
MAX_ITERS = 100

start = time.time()
# srcdata =tf.random_uniform([N,2])

# points = tf.Variable()
points =tf.Variable(conjunto_puntos)
cluster_assignments = tf.Variable(tf.zeros([N], dtype=tf.int64))

# Silly initialization:  Use the first K points as the starting
# centroids.  In the real world, do this better.
centroids = tf.Variable(tf.slice(points.initialized_value(), [0,0], [K,2]))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Replicate to N copies of each centroid and K copies of each
# point, then subtract and compute the sum of squared distances.
rep_centroids = tf.reshape(tf.tile(centroids, [N, 1]), [N, K, 2])
rep_points = tf.reshape(tf.tile(points, [1, K]), [N, K, 2])
sum_squares = tf.reduce_sum(tf.square(rep_points - rep_centroids),
                            reduction_indices=2)

# Use argmin to select the lowest-distance point
best_centroids = tf.argmin(sum_squares, 1)
did_assignments_change = tf.reduce_any(tf.not_equal(best_centroids,
                                                    cluster_assignments))

def bucket_mean(data, bucket_ids, num_buckets):
    total = tf.unsorted_segment_sum(data, bucket_ids, num_buckets)
    count = tf.unsorted_segment_sum(tf.ones_like(data), bucket_ids, num_buckets)
    return total / count

means = bucket_mean(points, best_centroids, K)

# Do not write to the assigned clusters variable until after
# computing whether the assignments have changed - hence with_dependencies
with tf.control_dependencies([did_assignments_change]):
    do_updates = tf.group(
        centroids.assign(means),
        cluster_assignments.assign(best_centroids))

changed = True
iters = 0

while changed and iters < MAX_ITERS:
    iters += 1
    [changed, _] = sess.run([did_assignments_change, do_updates])

[centers, assignments] = sess.run([centroids, cluster_assignments])
end = time.time()
print ("Found in %.2f seconds" % (end-start)), iters, "iterations"
print "Centroids:"
print centers
print "Cluster assignments:", assignments
res={"x":[],"y":[],"kmeans_res":[]}
for i in xrange(len(assignments)):
    res["x"].append(conjunto_puntos[i][0])
    res["y"].append(conjunto_puntos[i][1])
    res["kmeans_res"].append(assignments[i])
pd_res=pd.DataFrame(res)
sns.lmplot("x","y",data=pd_res,fit_reg=False,size=5,hue="kmeans_res")
plt.show()