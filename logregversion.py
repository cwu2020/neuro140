# conda create -n tensorflow_env tensorflow
# conda activate tensorflow_env
# may have to run 
# conda install pandas
# conda install scikit-learn 
# conda install -c conda-forge xgboost 

from sklearn.model_selection import train_test_split
from sklearn import linear_model
import tensorflow as tf
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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

log_model = linear_model.LogisticRegression(solver="liblinear")
log_model.fit(X = x_train, y = y_train)
print(log_model)
print(log_model.coef_)
preds = log_model.predict_proba(X = x_train)
predsdf = pd.DataFrame(preds)

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
predsdf['result'] = [0 if x>.5 else 1 for x in predsdf[0]]

# print("training accuracy", accuracy_score(y_train, predsdf['result']))

preds = log_model.predict_proba(X = x_test)
predsdf = pd.DataFrame(preds)

from sklearn.metrics import accuracy_score

predsdf['result'] = [0 if x >.5 else 1 for x in predsdf[0]]

print("test accuracy", accuracy_score(y_test, predsdf['result']))