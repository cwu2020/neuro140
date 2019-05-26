# conda create -n tensorflow_env tensorflow
# conda activate tensorflow_env
# may have to run 
# conda install pandas
# conda install scikit-learn 
# conda install -c conda-forge xgboost 

# separately each time bc for some reason conda isn't including it automatically in the environment for me

from sklearn.model_selection import train_test_split
from sklearn import linear_model
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# from keras.models import Sequential
# from keras.layers import Dense
# import matplotlib.pyplot as plt

df = pd.read_csv("/Users/carrawu/Documents/harvard/neuro140/confused_student_EEG/eeg-brain-wave-for-confusion/input/EEG_data_averaged_nosub6.csv")

# student independent classifier

seed=7
def set_aside_test_data(d):
	userdefinedlabel=d.pop("user-definedlabel") # pop off labels to new group, we'll be using userdefined label as our correct classification
	predefinedlabel=d.pop("predefinedlabel")
	subID=d.pop("SubjectID")
	vidID=d.pop("VideoID")
	# print(d)
	x_train,x_test,y_train,y_test = train_test_split(d,userdefinedlabel,test_size=0.2,random_state=seed)
	return x_train,x_test,y_train,y_test
	
x_train,x_test,y_train,y_test = set_aside_test_data(df)

print(x_train.shape)
def prep_test_data_forTF(x,y):
    x_test=x.values
    y_test=pd.get_dummies(y)
    y_test=y_test.values
    return x_test, y_test

x_test_TF, y_test_TF = prep_test_data_forTF(x_test,y_test)

batch_size=128
def get_mini_batch(x,y):
	rows=np.random.choice(x.shape[0], batch_size)
	return x[rows], y[rows]

def prep_train_data_forTF(x,y):
    x_train=x.values
    y_train=pd.get_dummies(y)
    y_train=y_train.values
    return x_train, y_train

x_train_numpy, y_train_numpy = prep_train_data_forTF(x_train,y_train)


sess = tf.Session()
lr=.0001
def trainNN(x_train_numpy, y_train_numpy,x_test_TF,y_test_TF,number_trials,number_nodes):
	# there are 11 features
	# place holder for inputs. feed in later
	print(x_train_numpy.shape)
	print(x_train.shape)
	x = tf.placeholder(tf.float32, [None, x_train_numpy.shape[1]])
	#weights
	w1 = tf.Variable(tf.random_normal([x_train_numpy.shape[1], 2],stddev=.5,name='w1'))
	#bias
	b1 = tf.Variable(tf.zeros([2]))
	# hidden_output = tf.nn.softmax(tf.matmul(x, w1) + b1)
	# w2 = tf.Variable(tf.random_normal([number_nodes, y_train_numpy.shape[1]],stddev=.5,name='w2'))
	# b2 = tf.Variable(tf.zeros([y_train_numpy.shape[1]]))
	# placeholder for correct values 
	y_ = tf.placeholder("float", [None,y_train_numpy.shape[1]])
	# These are predicted ys
	y = tf.nn.softmax(tf.matmul(x, w1) + b1)
	# Softmax activation function.
	# loss = tf.reduce_mean(tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits_v2(y, y_, name='xentropy')))
	# Sigmoid activation function
	loss = tf.reduce_mean(tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_, name='sigmoidXentropy')))
	opt = tf.train.GradientDescentOptimizer(learning_rate=lr)
	train_step = opt.minimize(loss, var_list=[w1,b1])
	# init all vars
	init = tf.global_variables_initializer()
	sess.run(init)
	ntrials = number_trials
	for i in range(ntrials):
	    # get mini batch
	    a,b=get_mini_batch(x_train_numpy,y_train_numpy)
	    # run train step, feeding arrays of 100 rows each time
	    _, cost =sess.run([train_step,loss], feed_dict={x: a, y_: b})
	    if i%1000 ==0:
	    	print("epoch is {0} and cost is {1}".format(i,cost))
	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print("test accuracy is {}".format(sess.run(accuracy, feed_dict={x: x_test_TF, y_: y_test_TF})))

trainNN(x_train_numpy,y_train_numpy,x_test_TF,y_test_TF,5000,5)



