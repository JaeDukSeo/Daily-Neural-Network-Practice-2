import numpy as np
import random
np.random.seed(67)

ground_truth = np.array([0.2,0.3,0.6,0.9,0.1])

lever_count = np.zeros((5))
expected =    np.zeros((5))
random_explore = 0.33

for i in range(1,201):

    # explore or greedy
    if random_explore > np.random.uniform():
        action = np.random.randint(0,5)
    else:
        action = np.argmax(expected)

    # make that action happen
    if random.random() < ground_truth[action]:
        reward = 1
    else:
        reward = 0

    lever_count[action] = lever_count[action] + 1
    expected[action] = expected[action] - (1/lever_count[action]) * (reward - expected[action]) 

        
print('Count for number of times each lever have been pulled : ', lever_count)
print('Double Check : ', lever_count.sum())
print('Agents Guess : ', np.around(expected/lever_count,2))
print('Ground Truth : ', ground_truth)

# -- end code --
