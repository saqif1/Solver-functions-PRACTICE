import numpy as np
import math
def A3_A0218299N(learning_rate, num_iters):
    eta = learning_rate
    max_iter = num_iters
    ### A3(a) - f(a) = a^4
    ## Step 1: Declare initialisation
    a = 2#CUSTOMISE THIS
    ## Step 2: Cost-function capture mechanism --> Empty arrays of 0
    a_out = np.zeros(max_iter)
    f1_out = np.zeros(max_iter)
    ## Step 3: Gradient Descent
    for i in range(0, max_iter):
        a = a - eta*4*(a**3) #CUSTOMISE THIS
        a_out[i] = a
        f1_out[i] = a ** 4 #CUSTOMISE THIS*OPTIONAL
    print(a_out)
    return a_out, f1_out

A3_A0218299N(.1,1) #CUSTOMISE THIS

