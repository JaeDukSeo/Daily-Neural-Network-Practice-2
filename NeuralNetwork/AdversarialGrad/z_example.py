import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,-1,1,4])
x2 = tf.constant([5,-6,-1,8])

temp_sess = tf.Session()
# Multiply
result = tf.multiply(x1, x2)
sss = tf.sign(x1)
sss1 = tf.sign(x2)

# Intialize the Session
sess = tf.Session()

# Print the result
print(sess.run(result))
print(sess.run(sss))
print(sess.run(sss1))

# Close the session
sess.close()


# -- end code -- 