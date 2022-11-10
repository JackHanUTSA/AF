
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from numpy import shape
import pandas as pd   
import csv 

# Size for each value
data_size = 5000
T = 10           # time interval

# initial condition
x0 = np.linspace(1,300,data_size)
# time space
t = np.linspace(0,T)
# t = 50 points


# make sure the T follow the interval

# parameter
k = 0.3
x0 = np.array(x0)

print('x0 = size',shape(x0))
print('t = size',shape(t))
print(t)

def model(x,t):
    dxdt = -k*x
    return dxdt
    
X = odeint(model,x0,t)
X = X.T             # reshape to t(coloum)*data_frame(row)

pd.DataFrame(X).to_csv("./ODE_300records.csv", header=None,index=None)

plt.plot(t,X[0,:],'b-',label=r'$x_0 = 1$')
plt.plot(t,X[2,:],'r-',label=r'$x_0 = 3$')
plt.plot(t,X[3000,:],'g-',label=r'$x_0 = 5$')
plt.xlabel('time')
plt.ylabel('x1(t)')
plt.show()