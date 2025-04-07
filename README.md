# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required library and read the dataframe.
2. Write a function computeCost to generate the cost function.
3. Perform iterations og gradient steps with learning rate.
4. Plot the Cost function using Gradient Descent and generate the required graph.

## Program:
```
/*
Program to implement the linear regression using gradient descent.
Developed by: Ashqar Ahamed S.T
RegisterNumber:  212224240018
*/
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
    x=np.c_[np.ones(len(x1)),x1]
    theta=np.zeros(x.shape[1]).reshape(-1,1)
    for _ in range(num_iters):
        predictions=(x).dot(theta).reshape(-1,1)
        errors=(predictions-y).reshape(-1,1)
        theta-=learning_rate*(1/len(x1))*x.T.dot(errors)
    return theta
data = pd.read_csv(r'C:\College\SEM 2\Machine Learning\Exp3\50_Startups.csv')
print(data.head())
x=(data.iloc[1:,:-2].values)
x1=x.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
x1_scaled=scaler.fit_transform(x1)
y1_scaled=scaler.fit_transform(y)

theta=linear_regression(x1_scaled,y1_scaled)
new_data=np.array([165249.2,136897.8,471784.1]).reshape(-1,1)
new_scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(f"prdiction value: {pre}")
```

## Output
![head](https://github.com/user-attachments/assets/50781859-bd6f-4c55-aead-a0033cf2f934)
## X Values:
![X values](https://github.com/user-attachments/assets/dc071045-5307-436f-9fb3-f65318427745)
## X Scaled Values:
![X1 values](https://github.com/user-attachments/assets/91712b7a-68eb-4b4b-9e8a-f739f92cb491)
## Predicted Values:
![Predict value](https://github.com/user-attachments/assets/0d6793de-7240-4e40-936b-3f0e8cf20988)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
