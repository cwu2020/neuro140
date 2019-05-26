# if you've already made the directory a tensorflow environment you can run just line 2
# conda create -n tensorflow_env tensorflow
# conda activate tensorflow_env
# may have to run 
# conda install pandas
# conda install scikit-learn 
# conda install -c conda-forge xgboost 

from sklearn.model_selection import train_test_split
from sklearn import linear_model
import tensorflow as tf
import xgboost
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import accuracy_score

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


model = xgboost.XGBClassifier()
model.fit(x_train, y_train)
print(model)
y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]
accuracy = accuracy_score(y_test, predictions)
print("student-independent classifier: ", accuracy)

# student specific classifier
df2 = pd.read_csv("/Users/carrawu/Documents/harvard/neuro140/confused_student_EEG/eeg-brain-wave-for-confusion/input/EEG_data_averaged_nosub6.csv")
accuracies = 0
seed=7
def set_aside_test_data(d):
	userdefinedlabel=d.pop("user-definedlabel") # pop off labels to new group, we'll be using userdefined label as our correct classification
	predefinedlabel=d.pop("predefinedlabel")
	subID=d.pop("SubjectID")
	vidID=d.pop("VideoID")
	# print(d)
	x_train,x_test,y_train,y_test = train_test_split(d,userdefinedlabel,test_size=0.2,random_state=seed)
	return x_train,x_test,y_train,y_test

for sub in range(9):
	studdata = df2.loc[sub*10:(sub+1)*10-1]
	# print(studdata)
	x_train,x_test,y_train,y_test = set_aside_test_data(studdata)
	model = xgboost.XGBClassifier()
	model.fit(x_train, y_train)
	# print(model)
	y_pred = model.predict(x_test)
	predictions = [round(value) for value in y_pred]
	accuracy = accuracy_score(y_test, predictions)
	accuracies += accuracy
	# print("xgboost accuracy: ", accuracy)
	# print(accuracy)
print("student-specific classifier: ", accuracies/9)
