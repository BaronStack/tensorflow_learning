#Program:
#   This program is a fully connected Neural Network to implement the classification. And also the data is from the mnist.
#History:
#2018-01-19     BaronStack      FirstRelease
#Result:
#I have test the different activation and the optimiser parameter that the best accuracy is :0.91...


from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None,):
    # add one more layer and return the output of this layer
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]),name = 'w')
            tf.summary.histogram(layer_name + '/weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.1,name = 'b')
            tf.summary.histogram(layer_name + '/biases',biases)
        with tf.name_scope('weights_plus_biases'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b,)
        tf.summary.histogram(layer_name + '/outputs',outputs)
        return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result

# define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, 784]) # 28x28
    ys = tf.placeholder(tf.float32, [None, 10])

# add output layer
prediction = add_layer(xs, 784, 10, n_layer=1, activation_function=tf.nn.softmax)

# the error between prediction and real data
with tf.name_scope('loss'):
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
    tf.summary.scalar('loss',cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.4).minimize(cross_entropy)

sess = tf.Session()

merged = tf.summary.merge_all()

writer = tf.summary.FileWriter("logs/", sess.graph)
# important step
# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()
sess.run(init)

for i in range(10000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 500 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))
        #result = sess.run(merged,feed_dict={xs: batch_xs, ys: batch_ys})
        #writer.add_summary(result, i)