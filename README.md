# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import Necessary Libraries: Import NumPy, pandas, and StandardScaler for numerical operations, data handling, and feature scaling, respectively.

2.Define the Linear Regression Function: Create a linear regression function using gradient descent to iteratively update parameters, minimizing the difference between predicted and actual values.

3.Load and Preprocess the Data: Load the dataset, extract features and target variable, and standardize both using StandardScaler for consistent model training.

4.Perform Linear Regression: Apply the defined linear regression function to the scaled features and target variable, obtaining optimal parameters for the model.

5.Make Predictions on New Data: Prepare new data, scale it, and use the trained model to predict the target variable, transforming predictions back to the original scale.

6.Print the Predicted Value 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Rogith J
RegisterNumber:  212224040280
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
dataset=pd.read_csv("Placement_Data.csv")
dataset

#dropping the serial no and salary col
dataset=dataset.drop("sl_no",axis=1)
dataset=dataset.drop("salary",axis=1)

dataset["gender"]=dataset["gender"].astype('category')
dataset["ssc_b"]=dataset["ssc_b"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["workex"]=dataset["workex"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset.dtypes

#labelling the columns
dataset["gender"]=dataset["gender"].cat.codes
dataset["ssc_b"]=dataset["ssc_b"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["workex"]=dataset["workex"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset

#selecting the features and labels
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values
#display independent variable
Y

#initialize the model parameters
theta=np.random.randn(X.shape[1])
y=Y
#define the sigmoid function
def sigmoid(z):
    return 1/(1+np.exp(-z))
#define the loss function
def loss(theta,X,y):
    h=sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))

#defining the gradient descent algorithm.
def gradient_descent(theta,X,y,alpha,num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h-y)/m
        theta -= alpha*gradient
    return theta
#train the model
theta = gradient_descent(theta,X,y,alpha=0.01,num_iterations=1000)
#makeprev \dictions
def predict(theta,X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h>=0.5,1,0)
    return y_pred
y_pred = predict(theta,X)


accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy:",accuracy)
print(Y)

xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)

xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## Output:
<img width="1236" height="449" alt="324679009-d10b3235-91f1-4915-9da0-9c71936ab8d1" src="https://github.com/user-attachments/assets/29c9d7c2-339f-44f4-ba24-670760d21998" />

<img width="755" height="323" alt="324679060-0fe9d8ba-c75f-495b-80c4-58899ece3e94" src="https://github.com/user-attachments/assets/9f6ff1f9-4312-4736-8c26-89875987f499" />


<img width="1191" height="450" alt="324679192-34b79650-e483-40c1-af47-2439c7982405" src="https://github.com/user-attachments/assets/a11dfcd4-2c7b-46f8-93b6-45ca587ee2bf" />


<img width="725" height="220" alt="324679231-f2de87a0-7588-4f72-831b-46a6f63f7ea4" src="https://github.com/user-attachments/assets/16f8c468-37c9-482d-a563-048f4c0fde92" />


<img width="314" height="36" alt="324679263-ee3d8ced-55ac-4ee4-bd87-64af79f39bfa" src="https://github.com/user-attachments/assets/6a798cd9-392c-458a-864a-473fef97d7a7" />


<img width="749" height="160" alt="324679286-c888db81-c5b3-4f4c-b535-9e1511171d74" src="https://github.com/user-attachments/assets/94220bf4-a646-4de2-99bf-5ffb0da47978" />

<img width="377" height="50" alt="324679307-84223e29-e40b-4cbf-968c-2f07f04aa2a6" src="https://github.com/user-attachments/assets/14b320a4-d862-4e24-b514-5fb892652439" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

