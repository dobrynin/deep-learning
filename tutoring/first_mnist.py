from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])

W1 = tf.Variable(tf.truncated_normal([784, 392]) / 780)
b1 = tf.Variable(tf.zeros([392]))
W2 = tf.Variable(tf.truncated_normal([392, 10]) / 392)
b2 = tf.Variable(tf.zeros([10]))

z1 = tf.matmul(x, W1) + b1
a1 = tf.nn.relu(z1)
z2 = tf.matmul(a1, W2) + b2

class_probabilities = tf.nn.softmax(z2)

true_class = tf.placeholder(tf.float32, [None, 10])

cross_entropies = tf.reduce_sum(true_class * -tf.log(class_probabilities), reduction_indices=[1])

average_cross_entropy = tf.reduce_mean(cross_entropies)

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(average_cross_entropy)

sess = tf.InteractiveSession()

sess.run(tf.global_variables_initializer())

for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, true_class: batch_ys})

correct_prediction = tf.equal(tf.argmax(true_class, 1), tf.argmax(class_probabilities, 1))

binary_correct_prediction = tf.cast(correct_prediction, tf.float32)

accuracy = tf.reduce_mean(binary_correct_prediction)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, true_class: mnist.test.labels}))
