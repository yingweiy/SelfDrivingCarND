import tensorflow as tf

# Solution is available in the other "solution.py" tab
import tensorflow as tf

# TODO: Convert the following to TensorFlow:
x = tf.constant(10)
y = tf.constant(2)
b = tf.constant(1)
z = tf.subtract(tf.divide(tf.cast(x, tf.float32), tf.cast(y, tf.float32)), tf.cast(b, tf.float32))

# TODO: Print z from a session
with tf.Session() as sess:

    # TODO: Feed the x tensor 123
    output = sess.run(z)
    print(output)
