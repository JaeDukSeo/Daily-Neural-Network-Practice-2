import numpy as np
import sys

np.set_printoptions(3,suppress=True)
np.random.seed(677)

def log_sig(x): return 1 / (1 + np.exp(-x))
def d_log_sig(x): return log_sig(x) * ( 1 - log_sig(x))

ground_truth_prob = np.random.rand(10)
number_of_levers_count = np.zeros(10)
agents_prob  = np.zeros(10)
reward_count = np.zeros(10)

num_episode = 10000
e = 0.99
learning_rate = 0.00000000001

class FNN():
    
    def __init__(self,inc,outc):
        self.w = np.random.randn(inc,outc) 

    def feed(self, input):
        self.input = input
        self.layer = np.dot(self.input,self.w)
        self.layerA = log_sig(self.layer)
        return self.layerA
    
    def back(self, grad):
        grad_part_1 = grad
        grad_part_2 = d_log_sig(self.layer)
        grad_part_3 = self.input

        grad_middle = grad_part_1 * grad_part_2
        grad = grad_part_3.T.dot(grad_middle)
        grad_pass = grad_middle.dot(self.w.T)
        self.w = self.w - learning_rate * grad
        return grad_pass

l1 = FNN(10,30)
l2 = FNN(30,10)

for _ in range(num_episode):
    
    # Either choose a greedy action or explore
    if e > np.random.uniform():
        which_lever_to_pull = np.random.randint(0,10)
    else:
        which_lever_to_pull = np.argmax(agents_prob)

    temp = np.zeros((1,10))
    temp[0,which_lever_to_pull] = 1
    layer1 = l1.feed(temp)
    layer2 = l2.feed(layer1)
    which_lever_to_pull2 = np.argmax(layer2)

    # now pull the lever 
    if ground_truth_prob[which_lever_to_pull2] > np.random.uniform():
        reward = 1
    else:
        reward = 0

    # now update the lever count and expected value
    number_of_levers_count[which_lever_to_pull2] += 1
    loss =  - np.log( np.clip(agents_prob * temp,1e-10,1e10)  ) * reward 
    back_prop = (-1/ np.clip(agents_prob * temp,1e-10,1e10)  )* reward
    grad2 = l2.back(back_prop)
    grad1 = l1.back(grad2)
    agents_prob = agents_prob - learning_rate * grad1 
    reward_count[which_lever_to_pull2] = reward_count[which_lever_to_pull2] + reward
    e = e * 0.9999

print('\n-------------------------')
print("Number of Levers Count Each: ", number_of_levers_count)
print("Sum of lever must match with # episode : ",number_of_levers_count.sum(), ' : ', num_episode)

print('\n-------------------------')
print("Reward Over Time Tracking : ", reward_count)
print("Agent's weight : ", agents_prob)
print("Agent's Probability Guess : ", reward_count/(number_of_levers_count+1e-10))
print("Agents Guess Best Lever : ", np.argmax(reward_count/(number_of_levers_count+1e-10)))

print('\n-------------------------')
print("Probability of each lever Ground Truth Full list : ", ground_truth_prob)
print("Ground Truth Best Lever : ", np.argmax(ground_truth_prob))

# -- end code --