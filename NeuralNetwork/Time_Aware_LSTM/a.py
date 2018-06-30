import tensorflow as tf
import numpy as np, datetime
from matplotlib import pyplot as plt
from cryptory import Cryptory
import pandas as pd,sys,os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.set_random_seed(678)
np.random.seed(567)

# activation and discount
def tf_sigmoid(x): return tf.sigmoid(x)
def tf_tanh(x): return tf.nn.tanh(x)
# def tf_discount(x): return 1 / tf.log(x + 1e-10)
def tf_discount(x): return 1 /x

# Class Time LSTM
class Time_LSTM():
    
    def __init__(self,timestamp,inc,outc):
        self.c = tf.Variable(tf.zeros(shape=[timestamp+1,1,outc]))
        self.h = tf.Variable(tf.zeros(shape=[timestamp+1,1,outc]))

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
#         (btc_google[['bitcoin', 'close']]-btc_google[['bitcoin', 'close']].min()))/
#         (btc_google[['bitcoin', 'close']].max()-btc_google[['bitcoin', 'close']].min())
# btc_google = btc_google.iloc[::-1]
# btc_google.to_csv('temp.csv')

# data from csv
df = pd.read_csv('temp.csv')
df = df[['bitcoin','close']]

# shift the bitcoin price by one to create label data
df['bitcoin_shift'] = df.bitcoin.shift(1)
df['search_shift']  = df.close.shift(1)

# split the data
bitcoin_label  = df.bitcoin_shift.values
search_label   = df.search_shift.values
bitcoin        = df.bitcoin.values
search         = df.close.values

# for label data drop the first value and for test data drop the last value
bitcoin_label = bitcoin_label[1:]
search_label = search_label[1:]
bitcoin  = bitcoin[:-1]
search  = search[:-1]

# merge the bitcoin and search data
data_merge = np.vstack((bitcoin,search)).T
label_merge = np.vstack((bitcoin_label,search_label)).T
print(data_merge.shape)
print(label_merge.shape)

# hyper 
num_epoch = 2
testing_days = 15
window_size = 4
learning_rate = 0.00001

# leave the last 50 data for test set
train_batch = data_merge[:-testing_days,:]
train_label = label_merge[:-testing_days]
test_batch =  data_merge[-testing_days:,:]
test_label =  label_merge[-testing_days:]

print(train_batch.shape)
print(train_label.shape)
print(test_batch.shape)
print(test_label.shape)

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

x2 = tf.convert_to_tensor(update_1)
update_2,update_2_w = [],[]
for current_range in range(window_size):
    layer_2,layer_2_w = l2.feed(x2[current_range],current_range,1)
    update_2.append(layer_2)
    update_2_w.append(layer_2_w)

weight_updates = update_1_w + update_2_w
final_output = tf.squeeze(tf.convert_to_tensor(update_2))
cost = tf.reduce_mean(tf.square(final_output-y))
auto_train = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# session
print('----------------------------')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    predicted_values = []

    for current_test_index in range(testing_days):

        # For number of epoch train the network using the sliding window
        for iter in range(num_epoch):
            for current_window_inex in range(len(train_batch)-window_size):
                current_window_data  = np.expand_dims(train_batch[current_window_inex:current_window_inex+window_size,:],1)
                current_window_label = train_label[current_window_inex:current_window_inex+window_size,0]
                sess_results = sess.run([cost,weight_updates,auto_train],feed_dict={x:current_window_data,y:current_window_label})
                print('Current Test Day: ',current_test_index, " Current Iter: ",iter," Current Cost: ", sess_results[0],end='\r')

        # get the last three day value and make the prediction
        current_prediction =np.expand_dims(train_batch[-window_size:,:],1)
        sess_reuslts = sess.run([final_output,weight_updates],feed_dict={x:current_prediction})
        sess_reuslts = sess_reuslts[0]
        # append the predicted value, we only want the last one
        predicted_values.append(sess_reuslts[-1])
        print('\n---------------------')
        print('Prediction for day: ', current_test_index, ' predicted value: ',predicted_values[-1], " Ground Truth Value : ",train_label[:1,0] )
        print('---------------------\n')

        # remove the first train data and append the first data from the label
        train_batch = np.vstack((train_batch[1:],train_label[:1]))
        # print(train_batch.shape)

        # remove the first train label and append the first data from the label
        train_label = np.vstack((train_label[1:],test_label[:1]))
        # print(train_label.shape)

    predicted_values = np.asarray(predicted_values)
    predicted_values = (predicted_values-predicted_values.min())/(predicted_values.max()-predicted_values.min())
    gt_values = test_label[:,0]
    gt_values = (gt_values-gt_values.min()) / (gt_values.max()-gt_values.min())

    plt.plot(predicted_values,color='red')
    plt.plot(gt_values,color='blue')
    plt.show()

    

        




# -- end code --