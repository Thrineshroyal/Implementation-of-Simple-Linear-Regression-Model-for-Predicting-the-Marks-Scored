# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: T.Thrinesh Royal
RegisterNumber:  212223230226
import pandas as pd
from sklearn import linear_model
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df.head())
print(df.tail())
reg=linear_model.LinearRegression()
x=df.iloc[:,:-1].values
y=df.iloc[:,1].values
print("X-Values",x)
print("Y-Values",y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
reg=linear_model.LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
print("Predicted values",y_pred)
print("Tested values",y_test)
*/
```
## Output:
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742402/2259351e-6a9d-4a62-b0fd-ea854e84f168)

```
plt.scatter(x_train,y_train,color="blue")
plt.plot(x_train,reg.predict(x_train),color="red")
plt.title("Hours vs Scores (Training_Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```
## Output:
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742402/e9925c91-d8fc-4cbd-9b06-2b6d0c97234c)



```
plt.scatter(x_test,y_test,color="black")
plt.plot(x_test,reg.predict(x_test),color="green")
plt.title("Hours vs Scores (Testing_Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:
![image](https://github.com/Jeshwanthkumarpayyavula/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145742402/9e10c01a-b7c2-4387-b0ef-bc9c0b0a3553)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
