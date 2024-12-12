# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 14:38:11 2023

@author: Annika Deutsch
@date: 9/9/2023
@title: HW 3
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

# imports
from scipy.special import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# define our bessel functions and plot them 
xj = np.linspace(0.0, 20.0, 500)
xy = np.linspace(0.75, 20.0, 500)
x = np.linspace(0.65, 20.0, 500)

fjx0 = j0(xj)
fjx1 = j1(xj)
fjx2 = jn(2,xj)
fyx0 = y0(xy)
fyx1 = y1(xy)
fyx2 = yn(2,xy)

def bessel(x):
    func = j0(x)*y0(x) - jn(2,x)*yn(2,x)
    return func

# bessel functions of the first kind
plt.figure(1)
plt.plot(xj, fjx0, 'b-', label='j0')
plt.plot(xj, fjx1, 'r--', label='j1')
plt.plot(xj, fjx2, 'm-.', label='j2')
plt.title('Bessel Functions of the First Kind')
plt.legend()

# bessel function of the second kind
plt.figure(2)
plt.plot(xy, fyx0, 'b-', label='y0')
plt.plot(xy, fyx1, 'r--', label='y1')
plt.plot(xy, fyx2, 'm-.', label='y2')
plt.title('Bessel Functions of the Secondx Kind')
plt.legend()

# bessel function expression we are evaluating the roots of 
plt.figure(3)
plt.plot(x, bessel(x))

####
# write a function to determine an acceptable bracket for a root
####
def bracket(func, x1, x2):
    """
    Given a function and an initial guessed range (x1,x2), expand the range
    until a root is bracketed in the returned values x1 and x2, or until 
    the range becomes unacceptably large
    """
    ntry = 20 
    factor = 1.5
    
    if x1==x2:
        raise Exception("Bad initial range")
    
    f1 = func(x1)
    f2 = func(x2)
    
    for i in range(0, ntry):
        if f1*f2 < 0.0:
            return (x1, x2)
        if abs(f1) < abs(f2): 
            x1 = x1 + factor*(x1-x2)
            f1 = func(x1)
        else:
            x2 = x2 + factor*(x1-x2)
            f2 = func(x2)
    return (x1, x2)
    
####
# write a function that finds a root via the bisection method
####

def bisection(func, x1, x2, acc):
    """
    Using bisection, return the root of a function to some desired accuracy, 
    and how many function evaluations are required to achieve such accuracy
    """
    nmax = 1000
    f = func(x1)
    fmid = func(x2)
    if f*fmid >= 0.0:
        raise Exception("Root must be bracketed for bisection to be" 
                        +" successful")

    if f < 0.0:
        root = x1
        dx = x2 - x1
    else:
        root = x2
        dx = x1 - x2
    
    for i in range(0, nmax):
        dx = dx * 0.5
        xmid = root + dx
        fmid = func(xmid)
        if fmid <= 0.0:
            root = xmid
        if abs(dx) < acc or fmid == 0.0:
            nevals = i + 1
            return root, nevals
    raise Exception("Too many bisections without finding root")
    
####    
# write a function that finds a root via the false position method
####

def false_pos(func, x1, x2, acc):
    """
    Using the false position method, return the root of a function to some 
    desired accuracy, as well as how many function evaluations are required
    to achieve such accuracy
    """
    nmax = 1000
    f1 = func(x1)
    f2 = func(x2)
    
    if f1*f2 >= 0.0:
        raise Exception("Root must be bracketed for bisection to be"+
                        "successful")
    
    for i in range(0, nmax):
        x3 = x1 - ((x2-x1)/(f2-f1)) * f1
        f3 = func(x3)
        
        if f3*f1 > 0.0:
            x1 = x3
            f1 = f3
        else:
            x2 = x3
            f2 = f3
     
        if abs(f3) < acc:
            nevals = i + 1 
            return x3, nevals
    raise Exception("Too many iterations without finding root")
        
####
#Find the first five positive roots of bessel() to an accuracy of 10**-7
####

# determine the brackets for the first five roots
x11, x21 = bracket(bessel, 0.65, 1.0)
x12, x22 = bracket(bessel, 1.0, 3.0)
x13, x23 = bracket(bessel, 3.0, 5.0)
x14, x24 = bracket(bessel, 5.0, 7.0)
x15, x25 = bracket(bessel, 7.0, 8.5)

x1 = [x11, x12, x13, x14, x15]
x2 = [x21, x22, x23, x24, x25]

# loop through the brackets and find the first five positive roots
acc = 10**-7
rootb = []
nevalsb = []
rootfp = []
nevalsfp = []
for i in range(0, 5):
    r1, n1 = bisection(bessel, x1[i], x2[i], acc)
    rootb.append(r1)
    nevalsb.append(n1)
    
    r2, n2 = false_pos(bessel, x1[i], x2[i], acc)
    rootfp.append(r2)
    nevalsfp.append(n2)
    

# save the results to a pandas dataframe 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
df = pd.DataFrame({
    'Bisection Roots': rootb,
    'Bisection Function Evals': nevalsb,
    'False Position Roots': rootfp,
    'False Position Function Evals': nevalsfp
})
print(df)

# plot the roots along with the expression whose roots we found
plt.figure(4)
plt.scatter(rootb, bessel(rootb), c='m', s=50)
plt.plot(x, bessel(x))

