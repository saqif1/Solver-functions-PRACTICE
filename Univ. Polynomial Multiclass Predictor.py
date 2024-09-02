#If we polynomial the train set, polynomial the test set as well!
##Trial Final Q14b
import numpy as np
from numpy.linalg import inv
from sklearn.preprocessing import PolynomialFeatures

##Step 1: Define training set
x_train = np.array( [[4,7,10,2,3,9]] )
order = 4 #customise what order
poly = PolynomialFeatures(order)
X_train = poly.fit_transform(x_train)
y_train = np.array( [[-1,-1,-1,1,1,1]] ) # class1 = [1,0,0] etc.

##Step 2: Train
def estimate_weights(P, y, reg=None):
    if reg == None: # least squares
        if P.shape[0] > P.shape[1]: #m(row) > d(column) -- Primal Ridge
            w = inv(P.T @ P) @ P.T @ y
        else: #-- Dual Ridge
            w = P.T @ inv(P @ P.T) @ y
    else:
        if P.shape[0] > P.shape[1]: #m(row) > d(column) -- Primal Ridge
            w = inv(P.T @ P + reg*np.eye(P.shape[1])) @ P.T @ y #P.shape[1] bcos P.T(dxn) @ P(nxd) = (dxd) *sliced from P
        else: #-- Dual Ridge
            w = P.T @ inv(P @ P.T + reg*np.eye(P.shape[0])) @ y #P.shape[0] bcos P(nxd) @ P.T(dxn) = (nxn) *sliced from P
    return w

w = estimate_weights(X_train,y_train)
#print(w)

##Step 3: Prediction
x_pred = np.array( [6] ).reshape(-1,1) #follow pythons advice
X_pred = poly.fit_transform(x_pred)
y_pred = X_pred @ w
#np.argmax(y_pred,axis=0) #to round up the local max
#print(np.argmax(y_pred,axis=1)+1) #because our class start from 1 not 0.
print(y_pred)
#print(np.shape(X_train))

