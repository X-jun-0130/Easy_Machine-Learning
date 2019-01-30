import tensorflow as tf
import numpy as np

train_x = []
train_y = []
f= open('LR_data.txt')
for line in f:
    eachline = line.strip().split()
    train_x.append([float(eachline[0]), float(eachline[1])])
    train_y.append(eachline[-1])
train_x = np.mat(train_x)
train_y = np.mat(train_y)
m, n = np.shape(train_x)

X = tf.placeholder(tf.float32, [m, n])
Y = tf.placeholder(tf.float32)

w = tf.Variable(tf.ones([n, 1]))
b = tf.Variable(-0.9)

layer_1 = tf.nn.sigmoid(tf.matmul(X, w)+b)
cost = -1.0/m*tf.reduce_sum(Y * tf.log(layer_1) + (1-Y)*tf.log(1-layer_1))

optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

maxEpochis = 20000


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epioch in range(maxEpochis):
        sess.run(optimizer, feed_dict = {X: train_x, Y: train_y})

        print(sess.run(cost, feed_dict = {X: train_x, Y: train_y}))

            #w  = sess.run(w, feed_dict = {X: train_x, Y: train_y})
            #print("Epoch:", epioch, "cost=", "{:.9f}".format(cost))

    #print(sess.run(w).flatten(), sess.run(b).flatten())


