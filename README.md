# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
import the standard libraries. Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. Import LabelEncoder and encode the dataset. Import LogisticRegression from sklearn and apply the model on the dataset. Predict the values of array. Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. Apply new unknown value

## Program:

```
NAME:KARTHICK RAJ M 
REG NO:212221040073
```

```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear") 
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]
````

## Output:

Original data(first five columns):

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/530d3f63-dd22-4fd7-b4df-b8b8e48ea23e)




Data after dropping unwanted columns(first five):




![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/c73f1172-5df2-4430-b34c-a54aa35bc5f7)


Checking the presence of null values:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/c5faf2e3-b523-4880-b603-86b699560d2a)


Checking the presence of duplicated values:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/f2cb54ed-7979-4158-8e74-f2389677541b)

Data after Encoding:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/2069a137-3db5-4406-a1ad-9b18aa0d11ad)





X DATA:



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/47ec0b69-2da3-49bb-b12d-eb65550df9e0)




Y DATA:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/aafe7add-38cc-427b-9ca1-7f88d4118a21)



Predicted Values:



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/d1a353da-8561-4407-926c-ad7e22e9c5c4)




Accuracy Score:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/a3960049-e013-47d3-9a3e-6889d4cdac93)



Confusion Matrix:


![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/6721f271-6643-4a6f-bc24-ee17e7897871)




Classification Report:

![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/b04aa699-cbbb-43c2-a5f1-7cf62fa3aad3)


Predicting output from Regression Model:



![image](https://github.com/KARTHICKRAJM84/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/128134963/64d00d8a-dd3b-4b1c-a6b2-72a994cca609)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
