#
# AEP 4380 HW 1, Annika Deutsch 
#
# Test numerical derivatives
#
# 
# Code adopted from E. Kirkland 21-aug-2023
#

import numpy as np
import matplotlib.pyplot as plt
#from math import * # can use math or numpy

#---- function to differentiate ------------
def feval(x):
    return(np.sin(x) * np.exp(-0.04*x*x))


n = 200 # number of points
xmin, xmax = -7.0, +7.0 # range of function
h = 0.5 # small number in limit
dx = (xmax - xmin)/(n-1) # step size of x
x = np.empty(n, float) # set size of x array
fx = np.empty(n, float) # set size of f(x)
fpfd = np.empty(n, float) # set size of forward derivatuve
fpbd = np.empty(n, float) # set size of backward derivative
fpcd = np.empty(n, float) # set size of central derivative 


#---- calculate curves with finite difference derivatives
for i in range(n):
    x[i] = xmin + i * dx
    fx[i] = feval(x[i])
    
    # forward derivative 
    fpfd[i] = (feval(x[i]+h) - fx[i])/h
    
    #backward derivative
    fpbd[i] = (fx[i] - feval(x[i]-h))/h
    
    # central derivative
    fpcd[i] = (feval(x[i]+h) - feval(x[i]-h))/(2*h)
    
    
#---- plot the functions and save in a file
plt.figure(1)
plt.plot(x, fx, 'k-', label='f')
plt.plot(x, fpfd, 'b--', label='fpfd')
plt.plot(x, fpbd, 'r--', label='fpbd')
plt.plot(x, fpcd, 'g--', label='fpcd')
plt.legend()
plt.xlabel( "x" )
plt.ylabel( "y" )
#plt.savefig('comp-phys\derivs.png')
plt.show()