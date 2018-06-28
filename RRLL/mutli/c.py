import numpy as np
import sys


np.set_printoptions(2)
# np.random.seed(67)


# activation fun
def atan(x): 
    return np.arctan(x)
def d_atan(x): 
    return 1/(1+atan(x) ** 2)

# class for neural network
class FNN():
    
    def __init__(self,in_c,out_c):
        self.w = np.random.randn(in_c,out_c) 

    def feedforward(self,input):
        self.input = input
        self.layer = np.dot(self.input, self.w)
        self.layerA = atan(self.layer)
        return self.layerA

    def back(self,grad):
        grad_1 = grad
        grad_2 = d_atan(self.layer)
        grad_3 = self.input

        grad_middle =  grad_1 * grad_2

        grad = grad_3.T.dot(grad_middle)
        grad_pass = grad_middle.dot(self.w.T)

        self.w = self.w - 0.000001 * grad

        return grad_pass 

l1 = FNN(10,15)
l2 = FNN(15,10)

ground_truth_prob = np.random.rand(10)
number_of_levers_count = np.zeros(10)
agents_prob = np.ones(10)
reward_count = np.zeros(10)

num_episode = 5000
e = 0.33
learning_rate = 0.001

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
        reward = -1

    # now update the lever count and expected value
    number_of_levers_count[which_lever_to_pull] += 1
    loss = -np.log( agents_prob[which_lever_to_pull] + 0.001  ) * reward 
    agents_prob[which_lever_to_pull] = agents_prob[which_lever_to_pull] - learning_rate * loss
    reward_count[which_lever_to_pull] = reward_count[which_lever_to_pull] + reward

print('\n-------------------------')
print("Number of Levers Count Each: ", number_of_levers_count)
print("Sum of lever must match with # episode : ",number_of_levers_count.sum(), ' : ', num_episode)

print("Reward Over Time Tracking : ", reward_count)
print("Agents Guess Best Lever : ", np.argmax(reward_count))

print("Ground Truth Full list : ", ground_truth_prob)
print("Ground Truth Best Lever : ", np.argmax(ground_truth_prob))

# -- end code --