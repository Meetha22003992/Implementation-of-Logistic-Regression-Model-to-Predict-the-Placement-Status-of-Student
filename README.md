# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Read the file "Placement.csv"
2. Copy the data of Placement.csv
3. Check for the null values in the data
4. Print the data status of the Placement.csv file
5. Print the X prediction array
6. Print the y prediction array
7. Get the accuracy value of the data
8. Print the confusion array 
9. Obtain the classification report
10. Ger the prediction of Logistic regression

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Meetha Prabhu
RegisterNumber:  212222240065
*/
**
import pandas as pd 
 
data=pd.read_csv('Placement_Data.csv')

data.head()
**
**
data1=data.copy()

data1= data1.drop(["sl_no","salary"],axis = 1)

data1.head()
**
**
data1.isnull().sum()
**
**
data1.duplicated().sum()
**
**
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
**
**
x=data1.iloc[:,:-1]
x
**
**
y=data1["status"]
y
**
**
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
**
**
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
**
**
from sklearn.metrics import confusion_matrix
confusion=(y_test,y_pred)
confusion
**
**
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
**

```

## Output:
![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/751f6b0d-6067-4f38-8665-2f12e78b74d7)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/33d88f22-92a4-4b48-ab89-38087dd354f0)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/c31e1153-0cb3-4fa8-995c-3d0b1d15d33d)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/156ba5b7-1257-4393-bead-0be1fe7e3a65)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/bc193cf7-e3f0-4634-8b89-a6b9d6ee264f)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/c6eeeded-719c-4366-8ad9-718228374f06)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/1ddca660-1873-4738-bf0a-8fa28a4ca273)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/57a6bb0a-6b1e-425d-80b8-0280a9a64250)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/3c2f9262-a68b-48af-b0ca-8ad80875b84b)

![image](https://github.com/Meetha22003992/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119401038/4dda99f2-ca15-4f8d-9665-549ebd041a48)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
