import random, datetime, math

random.seed(datetime.datetime.now())
levers = ["q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10"]
reward_percent = [random.random() for i in range(10)]
lever_count, reward_count, expected_return  = ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for i in range(3))

lever = random.randint(0, 9)
epsilon = 0.33
optimal_decision = reward_percent.index(max(reward_percent))

for i in range(1, 1001):
    #epislon case - explore if the random value is less than epsilon's value we choose a random lever
    if random.random() < epsilon:
        vals = list(range(0, 10))
        vals.remove(expected_return.index(max(expected_return)))
        lever = random.choice(vals)

    #if we havent received a reward yet and our epsilon case did not occur, just stick with the same lever (if we just use the case below, max() will return 0 on our first iteration)
    elif max(reward_count) == 0:
        lever = lever

    #else, pick the value that has historically given the highest reward
    else:
        lever = expected_return.index(max(expected_return))

    #every 100 iterations, print out the optimal lever, its chance and times. also print out the average reward
    if i % 100 == 0:
        print("Optimal Decision: %s with chance: %f chosen %d times. Average reward: %f" % (levers[optimal_decision], reward_percent[optimal_decision], lever_count[optimal_decision], sum(expected_return)))

    #pray to RNGesus that we get a reward from our current lever
    if random.random() < reward_percent[lever]:
        reward_count[lever]+=1
        instant_reward = 1
    else:
        instant_reward = 0

    #update how many times the current lever was pulled
    lever_count[lever]+=1

    #newestimate function
    expected_return[lever] = expected_return[lever] + (1/lever_count[lever])* (instant_reward - expected_return[lever])

#rounding at the end for prettier printing
reward_percent = [int((math.ceil(num*100)/100)*100) for num in reward_percent]
expected_return = [math.ceil(num*100)/100 for num in expected_return]

print("Epsilon: ", epsilon)
print("Times each lever was pulled: ", lever_count)
print("Percent chance for each lever to give reward: ", reward_percent)
print("Expected return value for each lever: ", expected_return)



import tensorflow as tf
import numpy as np
import random, datetime
random.seed(datetime.datetime.now())

bandits = ["q"+str(x) for x in range(1,11)]
no_of_bandits = len(bandits)

#start of neural network
tf.reset_default_graph()

#initializes the weights and sets to choose the action using argmax
weights = tf.Variable(tf.ones([no_of_bandits]))
chosen_action = tf.argmax(weights,0)

#Give reward value and action to the network to compute win or loss, train and update the network.
reward_holder = tf.placeholder(shape=[1],dtype=tf.float32)
action_holder = tf.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(tf.log(responsible_weight)*reward_holder)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.002)
update = optimizer.minimize(loss)

#array to see if the agent was correct or not:
score = [0 for i in range(len(bandits))]

for iterator in range(10):
    #create % chance for bandits to return reward and initialize array to hold amount of rewards from each bandit
    reward_percent = [random.random() for x in range(no_of_bandits)]
    total_reward = np.zeros(no_of_bandits)

    #choose first action
    lever = random.randint(0, 9)
    epsilon = 0.2222
    total_episodes = 1000 #Set total number of episodes to train agent on.

    #initialize neural net
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        i = 0
        while i < total_episodes:
            if random.random() < epsilon:
                vals = list(range(0, 10))
                vals.remove(sess.run(chosen_action))
                lever = random.choice(vals)

            #else, pick the value that has historically given the highest reward
            else:
                lever = sess.run(chosen_action)
            #pray to RNGesus that we get a reward from our lever
            reward = 0
            if random.random() < reward_percent[lever]:
                reward = 1
            else:
                reward = -1

            #update the total reward for the action we chose
            total_reward[lever] += reward
            #update our network
            _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[lever]})

            i+=1
    #if bandit with highest reward equals to bandit estimated to have highest reward, agent was correct
    if(np.argmax(reward_percent) == np.argmax(ww)):
        score[iterator] = "Correct"
    else:
        score[iterator] = "X"
print(score)