import gym,sys
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
env = gym.make('FrozenLake-v0')


class FNN():
    def __init__(self,input_dim,hidden_dim):
        self.w = tf.Variable(tf.random_normal([input_dim,hidden_dim],stddev=0.05))
        # self.w = tf.Variable(tf.random_uniform([input_dim,hidden_dim],0,0.01))
        # self.w = tf.Variable(xavier_init(input_dim,hidden_dim))

    def feedforward(self,input=None):
        self.input = input
        self.layer = tf.matmul(input,self.w)
        self.layerA = tf.nn.elu(self.layer)
        return self.layerA

# hyper
learning_rate = 0.1
q_table_learning_rate = 0.99
e = 0.1 
num_iter = 1000
max_episode = 1000

#create lists to contain total rewards and steps per episode
jList = []
rList = []

# classes
l1 = FNN(16,10)
l2 = FNN(10,4)

# place holders
x = tf.placeholder(shape=[1,16],dtype=tf.float32)
y = tf.placeholder(shape=[1,4],dtype=tf.float32)

layer1 = l1.feedforward(x)
layer2 = l2.feedforward(layer1)

next_action = tf.argmax(layer2,1)

loss = tf.reduce_sum(tf.square(y - layer2))
trainer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# sess
with tf.Session() as sess: 

    sess.run(tf.global_variables_initializer())

    for i in range(num_iter):
        
        state = env.reset()
        rAll = 0
        done = False

        for j in range(max_episode):
            
            #Choose an action by greedily (with e chance of random action) from the Q-network
            action,allQ = sess.run([next_action,layer2],
            feed_dict={x:np.identity(16)[state:state+1]})
            action = action[0]

            if np.random.rand(1) < e:
                action = env.action_space.sample()

            next_state,reward,done,_ = env.step(action)

            Q1 = sess.run(layer2,
            feed_dict={x:np.identity(16)[next_state:next_state+1]})

            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            targetQ = allQ
            targetQ[0,action] = reward + q_table_learning_rate*maxQ1

            #Train our network using target and predicted Q values
            _ = sess.run(trainer,feed_dict={
                x:np.identity(16)[state:state+1],y:targetQ})

            rAll += reward
            state = next_state
            if done == True:
                #Reduce chance of random action as we train the model.
                e = 1./((i/50) + 10)
                break
    
        jList.append(j)
        rList.append(rAll)

print("Percent of succesful episodes: " + str(sum(rList)/num_iter) + "%")
plt.plot(rList)
plt.show()
plt.plot(jList)
plt.show()

# --- end code ---