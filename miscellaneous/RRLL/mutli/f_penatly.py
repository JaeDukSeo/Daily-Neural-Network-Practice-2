import numpy as np
np.set_printoptions(2)

ground_truth_prob = np.random.rand(10)
number_of_levers_count = np.zeros(10)
agents_prob = np.ones(10) * 0.1
reward_count = np.zeros(10)

num_episode = 40000
e = 0.33
alpha = 0.2
beta = 0.8 

for _ in range(num_episode):
    
    # Either choose a greedy action or explore
    if e > np.random.uniform():
        which_lever_to_pull = np.random.randint(0,10)
    else:
        which_lever_to_pull = np.random.choice(10, p=agents_prob)

    # now pull the lever 
    if ground_truth_prob[which_lever_to_pull] > np.random.uniform():
        reward = 1
    else:
        reward = 0

    # now update the lever count and expected value
    number_of_levers_count[which_lever_to_pull] += 1
    reward_count[which_lever_to_pull] = reward_count[which_lever_to_pull] + reward
    mask1 = np.zeros(10)
    mask2 = np.ones(10)
    mask1[which_lever_to_pull] = 1
    mask2[which_lever_to_pull] = 0
    if reward == 1:
        lever_won_vector = (agents_prob + alpha * (1- agents_prob)) * mask1
        rest_of_levers   = ((1-alpha) * agents_prob) * mask2
        agents_prob = lever_won_vector + rest_of_levers
    else:
        rest_of_levers = ( beta / (len(ground_truth_prob)-1)  + (1-beta) *agents_prob ) * mask2
        lever_lose_vector = ( (1-beta)*agents_prob )* mask1
        agents_prob = lever_lose_vector + rest_of_levers


print('\n-------------------------')
print("Number of Levers Count Each: ", number_of_levers_count)
print("Sum of lever must match with # episode : ",number_of_levers_count.sum(), ' : ', num_episode)

print('\n-------------------------')
print("Reward Over Time Tracking : ", reward_count)
print("Agent's weight : ", agents_prob)
print("Agent's Probability Guess : ", reward_count/number_of_levers_count )
print("Agents Guess Best Lever : ", np.argmax(reward_count/number_of_levers_count))

print('\n-------------------------')
print("Ground Truth Full list : ", ground_truth_prob)
print("Ground Truth Best Lever : ", np.argmax(ground_truth_prob))

# -- end code --