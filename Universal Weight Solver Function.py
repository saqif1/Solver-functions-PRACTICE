from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv
import numpy as np

#### 1.Weight Solver
def estimate_weights(P, y, reg=None):
    if reg == None: # least squares
        if P.shape[0] > P.shape[1]: #m(row) > d(column) -- Primal Ridge #left
            w = inv(P.T @ P) @ P.T @ y
        else: #-- Dual Ridge
            w = P.T @ inv(P @ P.T) @ y #right
    else:
        if P.shape[0] > P.shape[1]: #m(row) > d(column) -- Primal Ridge
            w = inv(P.T @ P + reg*np.eye(P.shape[1])) @ P.T @ y #P.shape[1] bcos P.T(dxn) @ P(nxd) = (dxd) *sliced from P
        else: #-- Dual Ridge
            w = P.T @ inv(P @ P.T + reg*np.eye(P.shape[0])) @ y #P.shape[0] bcos P(nxd) @ P.T(dxn) = (nxn) *sliced from P
    return w

#### 2. Regressor Solver
def CreateRegressor(x,order):
    P = np.ones(len(x))
    for n in range(1,order+1):
        P=np.column_stack((P,x**n))
    return P

#print(CreateRegressor(np.array([[1,1],[0,1],[3,3]]),3))

### 3. How many parameter to learn?
z=np.array( [ [1,1],[0,1],[3,3],[4,6]] ) #to change
order = 2 #to change
poly=PolynomialFeatures(order)
P = poly.fit_transform(z)
print(np.shape(P))

