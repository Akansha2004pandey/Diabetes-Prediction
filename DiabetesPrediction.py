import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score


diabetes_dataset=pd.read_csv('diabetes.csv')
#pd.read_csv? while using google collaborator
#printing the first 5 rows
print(diabetes_dataset.head())
#1 diabetic
#2 non diabetic
#no of rows and columns
#number of rows and columns in this
print(diabetes_dataset.shape)
#getting the statistical measures of the data
print(diabetes_dataset.describe())
#it will give all the statistical measures of the data
print(diabetes_dataset['Outcome'].value_counts())
#0-->non diabetic people
#1-->diabetic people
print(diabetes_dataset.groupby('Outcome').mean())
#separating the data and labels 
X= diabetes_dataset.drop(columns='Outcome',axis=1)
#if droping a column axis =1
#if dropping a row  axis=0
Y=diabetes_dataset['Outcome']
print(X)
print(Y)
#in python indexing from 0
#data preprocessing standardize the data
#we are going to use standard scaler
scaler=StandardScaler()
scaler.fit(X)
standardized_data=scaler.transform(X)
#we can use scaler.transform
#to get all the values in similar range
X=standardized_data
print(X)
print(Y)
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y, random_state=2)
#all diabetic may be going to x_train and all non diabetic to test data hence we are using stratify to avoid that
print(X.shape,X_train.shape,X_test.shape)
classifier=svm.SVC(kernel='linear')
classifier.fit(X_train,Y_train)
prediction=classifier.predict(X_test)
accuracy=accuracy_score(prediction,Y_test)
print('Accuracy score of the training data:', accuracy)
#it is not overfitted or overtrained data
#Making a predictive system
input_data=(4,110,92,0,0,37.6,0.191,30)
#changing the input data to a numpy array 
input_data_as_numpy_array=np.asarray(input_data)
#reshape the array as we are predicting for one instance 
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# standardize the input data 
std_data=scaler.transform(input_data_reshaped)
print(std_data)
pred=classifier.predict(std_data)
print(pred)

if(pred==0):
    print("the person is non diabetic!")
else:
    print("person is diabetic!!!")



#Visulaizing the trianing set result  




