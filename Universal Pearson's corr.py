from sklearn.preprocessing import PolynomialFeatures
from numpy.linalg import inv
import numpy as np

### 4. Pearson's correlation solver - Choose highest absolute r
def pearsons_corr(x,y):
    mu_x = np.mean(x)
    mu_y = np.mean(y)
    cov_xy = (x-mu_x) @ (y-mu_y) / len(x) #Vector

    sigma_x = np.sqrt(np.mean((x - mu_x)**2)) #Number
    sigma_y = np.sqrt(np.mean((y - mu_y) ** 2)) #Number

    rho_xy =  cov_xy / (sigma_x * sigma_y)
    #print(rho_xy)
    return rho_xy

feature1 = np.array([-1.7253,-0.7804,-0.9944,0.5307,-1.0502 ]) #to customise
Target_y = np.array([ 2.9972,1.1399,2.228,0.3387,2.5042]) #to customise
print(pearsons_corr(feature1,Target_y))