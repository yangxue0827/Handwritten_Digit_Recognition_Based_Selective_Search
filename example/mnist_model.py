from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os
import time
import numpy as np

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
sess = tf.InteractiveSession()

learning_rate = 1e-4
iteration = 0
batch_size = 50
test_step = 100
dropout_pro = 0.5
model_path = './model/model.ckpt'


def time_shitf():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])

with tf.name_scope('conv1'):
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

with tf.name_scope('conv2'):
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

with tf.name_scope('fc1'):
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

with tf.name_scope('fc2'):
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('evaluation'):
    pre = tf.argmax(y_conv, 1)
    correct_prediction = tf.equal(pre, tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

tf.global_variables_initializer().run()

saver = tf.train.Saver()

if os.path.exists(model_path + '.index'):
    saver.restore(sess, model_path)
    print('%s:restore model...' % time_shitf())
else:
    print('%s:start training...' % time_shitf())

for i in range(iteration):
    batch = mnist.train.next_batch(batch_size)
    if (i % 100 == 0) or (i + 1 == iteration):
        batch_test = mnist.test.next_batch(batch_size * 2)
        train_accuracy, loss = sess.run((accuracy, cross_entropy),
                                        feed_dict={x: batch_test[0], y_: batch_test[1], keep_prob: 1.0})
        print("%s:step %d, training accuracy is %.4f, and loss is %.4f." % (time_shitf(), i, train_accuracy, loss))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: dropout_pro})

    if ((i % 1000 == 0) and (i > 0)) or (i + 1 == iteration):
        saver.save(sess, model_path)
        print('%s:save model...' % time_shitf())

        # image_data = np.load('./image_data.npy')
        # print('%s:the number you write is:', end='')
        # result = []
        # for j in range(len(image_data)):
        #     data = np.reshape(image_data[j], [-1, 784])
        #     num = sess.run(pre, feed_dict={x: data, keep_prob: 1})
        #     result.append(num)
        #     print(num, end='')

# test_accuracy, final_loss = sess.run((accuracy, cross_entropy),
#                                      feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
# print("%s:test accuracy is %.4f, and loss is %.4f." % (time_shitf(), test_accuracy, final_loss))

image_data = np.load('./data/image_data.npy')
print('the number you write is:', end='')
result = []
for i in range(len(image_data)):
    data = np.reshape(image_data[i], [-1, 784])
    num = sess.run(pre, feed_dict={x: data, keep_prob: 1})
    result.append(num)
    print(num, end='')

sess.close()

