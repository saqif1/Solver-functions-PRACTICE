import numpy as np
from numpy.linalg import inv

x_train = np.array([[1,3,-2],[-4,0,-1],[3,1,8],[2,1,6],[8,4,6]])
x_train_bias = np.column_stack((np.ones(len(x_train)),x_train))
y_train = np.array( [ [1,0,0],[1,0,0],[0,1,0],[0,0,1],[0,0,1] ] )

#print(x_train_bias)
w = inv(x_train_bias.T @ x_train_bias) @ x_train_bias.T @ y_train
#print(y_train)
x_pred = np.array([ [1,-2,4] ]) # be careful with [ [] [] [] ] make sure square bracket encompasses all
x_pred_bias = np.column_stack((np.ones(len(x_pred)),x_pred))
y_pred = x_pred_bias @ w
print(y_pred)