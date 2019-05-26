# conda create -n tensorflow_env tensorflow
# conda activate tensorflow_env
# may have to run 
# conda install pandas
# conda install scikit-learn 
# conda install -c conda-forge xgboost 

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import pandas as pd
import io
import math
from scipy import stats


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
def prep_data_forTF(x,y):
    x=x.values
    y=pd.get_dummies(y)
    y=y.values
    return x, y

x_train, y_train = prep_data_forTF(x_train,y_train)
x_test, y_test = prep_data_forTF(x_test,y_test)


# inputs
training_epochs = 3000
learning_rate = 0.01
hidden_layers = 10
cost_history = np.empty(shape=[1],dtype=float)

X = tf.placeholder(tf.float32,[None,x_train.shape[1]])
# print(y_train.shape)
Y = tf.placeholder(tf.float32,[None,y_train.shape[1]])
is_training=tf.Variable(True,dtype=tf.bool)


# models

initializer = tf.contrib.layers.xavier_initializer()
h0 = tf.layers.dense(X, hidden_layers, activation=tf.nn.relu, kernel_initializer=initializer)
# h0 = tf.nn.dropout(h0, 0.95)
h1 = tf.layers.dense(h0, y_train.shape[1], activation=None)

cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=h1)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# prediction = tf.argmax(h0, 1)
# correct_prediction = tf.equal(prediction, tf.argmax(Y_one_hot, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

predicted = tf.nn.sigmoid(h1)
correct_pred = tf.equal(tf.round(predicted), Y)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# session

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(training_epochs + 1):
        sess.run(optimizer, feed_dict={X: x_train, Y: y_train})
        loss, _, acc = sess.run([cost, optimizer, accuracy], feed_dict={
                                 X: x_train, Y: y_train})
        cost_history = np.append(cost_history, acc)
        if step % 500 == 0:
            print("Step: {:5}\tLoss: {:.3f}\tAcc: {:.2%}".format(
                step, loss, acc))
            
    # Test model and check accuracy
    print('Test Accuracy:', sess.run(accuracy, feed_dict={X: x_test, Y: y_test}))
    