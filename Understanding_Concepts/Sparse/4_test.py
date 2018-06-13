import tensorflow as tf

g1 = tf.Graph()
with g1.as_default() as g:
    with g.name_scope( "g1" ) as g1_scope:
        temp = tf.placeholder(shape=[],dtype=tf.float32,name='temp')
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.multiply( temp, matrix2, name = "product")

tf.reset_default_graph()

g2 = tf.Graph()
with g2.as_default() as g:
    with g.name_scope( "g2" ) as g2_scope:
        temp = tf.placeholder(shape=[],dtype=tf.float32,name='temp')
        matrix2 = tf.constant([[5.],[5.]])
        product = tf.multiply( temp, matrix2, name = "product" )

tf.reset_default_graph()

use_g1 = False

if ( use_g1 ):
    g = g1
    scope = g1_scope
else:
    g = g2
    scope = g2_scope

# temp = [g1,g2]

with tf.Session( graph = g ) as sess:
    tf.global_variables_initializer()
    temp = sess.graph.get_tensor_by_name( scope + "temp:0")
    result = sess.run( sess.graph.get_tensor_by_name( scope + "product:0" ),feed_dict={temp:0.8} )
    # result = sess.run( product )
    print( result )