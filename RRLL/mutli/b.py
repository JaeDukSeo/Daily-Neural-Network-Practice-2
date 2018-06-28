import numpy as np
np.set_printoptions(2)

ground_truth_prob = np.random.rand(10)
number_of_levers_count = np.zeros(10)
agents_prob = np.zeros(10)

num_episode = 100000
e = 0.33

for _ in range(num_episode):
    
    # Either choose a greedy action or explore
    if e > np.random.uniform():
        which_lever_to_pull = np.random.randint(0,10)
    else:
        which_lever_to_pull = np.argmax(agents_prob)

    # now pull the lever 
    if ground_truth_prob[which_lever_to_pull] > np.random.uniform():
        reward = 1
    else:
        reward = 0

    # now update the lever count and expected value
    number_of_levers_count[which_lever_to_pull] += 1
    agents_prob[which_lever_to_pull] = agents_prob[which_lever_to_pull] + (1/number_of_levers_count[which_lever_to_pull]) * \
                                       (reward - agents_prob[which_lever_to_pull])

print("Number of Levers Count Each: ", number_of_levers_count)
print("Sum of lever must match with # episode : ",number_of_levers_count.sum(), ' : ', num_episode)

print("Agents Guess Full list : ", agents_prob)
print("Agents Guess Best Lever : ", np.argmax(agents_prob))

print("Ground Truth Full list : ", ground_truth_prob)
print("Ground Truth Best Lever : ", np.argmax(ground_truth_prob))

# -- end code --