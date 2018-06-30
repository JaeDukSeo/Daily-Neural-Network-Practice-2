import tensorflow as tf
import numpy as np, datetime
from matplotlib import pyplot as plt
from cryptory import Cryptory
import pandas as pd,sys

tf.set_random_seed(678)
np.random.seed(567)

# activation and discount
def tf_sigmoid(x): return tf.sigmoid(x)
def tf_tanh(x): return tf.nn.tanh(x)
def tf_discount(x): return 1 / tf.log(x + 1e-10)

# Class Time LSTM
class Time_LSTM():
    
    def __init__(self,timestamp,inc,outc):
        self.c = tf.Variable(tf.random_normal(shape=[timestamp+1,1,outc],stddev=0.05))
        self.h = tf.Variable(tf.random_normal(shape=[timestamp+1,1,outc],stddev=0.05))

        self.w_forget = tf.Variable(tf.random_normal(shape=[inc,outc],stddev=0.05))
        self.u_forget = tf.Variable(tf.random_normal(shape=[outc,outc],stddev=0.05))

        self.w_input = tf.Variable(tf.random_normal(shape=[inc,outc],stddev=0.05))
        self.u_input = tf.Variable(tf.random_normal(shape=[outc,outc],stddev=0.05))

        self.w_output = tf.Variable(tf.random_normal(shape=[inc,outc],stddev=0.05))
        self.u_output = tf.Variable(tf.random_normal(shape=[outc,outc],stddev=0.05))

        self.w_c = tf.Variable(tf.random_normal(shape=[inc,outc],stddev=0.05))
        self.u_c = tf.Variable(tf.random_normal(shape=[outc,outc],stddev=0.05))

        self.w_decompose = tf.Variable(tf.random_normal(shape=[outc,outc],stddev=0.05))

    def feed(self,input,time,time_difference):
        
        c_s_t_1 = tf_tanh(tf.matmul(self.c[time,:],self.w_decompose))
        c_s_hat_t_1 = c_s_t_1*tf_discount(time_difference)

        c_t_difference = self.c[time,:] - c_s_t_1
        c_t_adjusted   = c_t_difference + c_s_hat_t_1

        forget_gate = tf_sigmoid(tf.matmul(input,self.w_forget) + 
                                 tf.matmul(self.h[time,:],self.u_forget))

        input_gate = tf_sigmoid(tf.matmul(input,self.w_input) + 
                                 tf.matmul(self.h[time,:],self.u_input))

        output_gate = tf_sigmoid(tf.matmul(input,self.w_output) + 
                                 tf.matmul(self.h[time,:],self.u_output))

        candidate = tf_tanh(tf.matmul(input,self.w_c) + 
                                 tf.matmul(self.h[time,:],self.u_c))

        currrent_memory = forget_gate * c_t_adjusted + input_gate * candidate
        currrent_hidden = output_gate * tf_tanh(currrent_memory)

        update_state = []

        update_state.append(tf.assign( self.c[time+1,:,:],  currrent_memory ))
        update_state.append(tf.assign( self.h[time+1,:,:],  currrent_hidden   ))

        return currrent_hidden,update_state

# data code from: https://github.com/dashee87/cryptory
# my_cryptory = Cryptory(from_date = "2017-02-14",to_date="2018-06-30")
# btc_google = my_cryptory.get_google_trends(kw_list=['bitcoin']).merge(
#     my_cryptory.extract_coinmarketcap('bitcoin')[['date','close']], on='date', how='inner')

# # need to scale columns (min-max scaling)
# btc_google[['bitcoin','close']] = (
#         btc_google[['bitcoin', 'close']]-btc_google[['bitcoin', 'close']].min())/(
#         btc_google[['bitcoin', 'close']].max()-btc_google[['bitcoin', 'close']].min()
#     )
# btc_google = btc_google.iloc[::-1]
# btc_google.to_csv('temp.csv')

# data from csv
df = pd.read_csv('temp.csv')
df = df[['bitcoin','close']]

# shift the bitcoin price by one to create label data
df['bitcoin_shift'] = df.bitcoin.shift(1)

# split the data
bitcoin_label  = df.bitcoin_shift.values
bitcoin        = df.bitcoin.values
search         = df.close.values

print(bitcoin_label[:5])
print(bitcoin[-5:])
print(search[-5:])

# for label data drop the first value and for test data drop the last value
bitcoin_label = bitcoin_label[1:]
bitcoin  = bitcoin[:-1]
search  = search[:-1]


print(bitcoin_label[:5])
print(bitcoin[-5:])
print(search[-5:])

sys.exit()

# seperate the data
bitcoin_prices = df.bitcoin.values
search         = df.close.values
dates          = df.date.values

temp_merge = np.vstack((bitcoin_prices,search)).T
print(temp_merge.shape)

train_batch = bitcoin_prices[:50]
# train_label = 
# test_batch = 
# test_label = 

print(train_batch.shape)
print(train_batch.shape)
print(train_batch.shape)
print(train_batch.shape)

sys.exit()
# hyper 
num_epoch = 100

window_size = 50
learning_rate = 0.001

#  define class
l1 = Time_LSTM(window_size,2,5)
l2 = Time_LSTM(window_size,5,1)

# graph
x = tf.placeholder(shape=[window_size,1,2],dtype=tf.float32)
y = tf.placeholder(shape=[window_size],dtype=tf.float32)

update_1,update_1_w = [],[]
for current_range in range(window_size):
    layer_1,layer_1_w = l1.feed(x[current_range],current_range,1)
    update_1.append(layer_1)
    update_1_w.append(layer_1_w)

input_layer_2 = tf.convert_to_tensor(update_1)
update_2,update_2_w = [],[]
for current_range in range(window_size):
    layer_2,layer_2_w = l2.feed(input_layer_2[current_range],current_range,1)
    update_2.append(layer_2)
    update_2_w.append(layer_2_w)

final_output = tf.squeeze(tf.convert_to_tensor(update_2))
cost = tf.reduce_mean(tf.square(final_output-y))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    # for iter in range(num_epoch):
        




# -- end code --