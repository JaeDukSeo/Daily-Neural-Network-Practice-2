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