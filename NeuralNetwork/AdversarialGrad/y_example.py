import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,-1,1,4])
x2 = tf.constant([5,-6,-1,8])

# Multiply
sss = tf.sign(x1)
sss1 = tf.sign(x2)

# Intialize the Session
sess = tf.Session()

# Print the result
print(dir(x1))
print("Input Array: ", sess.run(x1))
print("Return Sign Array: ",sess.run(sss))
print('---------------------')
print("Input Array: ", sess.run(x2))
print("Return Sign Array: ",sess.run(sss1))

# Close the session
sess.close()


# -- end code -- 