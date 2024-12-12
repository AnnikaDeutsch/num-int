# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 11:32:54 2023

@author: Annika Deutsch
@date: 9/27/2023
@title: HW 4
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

# imports
import numpy as np
from numba import njit 
import matplotlib.pyplot as plt
import rk4 #python file containing general ODE solver rk4()

#------------------test rk4 on SHO diff eqn---------------------

@njit
def rhs_SHO(t, y):
    """
    Parameters
    ----------
    t : int
        time step at which are evaluating the rhs.
    y : array
        y vector at that time, for which each of the N equations could be 
        dependent.

    Returns
    -------
    f : array
        1D NumPy array holding rhs func eval of each .
    
    """
    f = np.zeros(2)
    f[0] = y[1]
    f[1] = -1.0 * y[0] # take \omega = 1
    
    return f

nt = 500
neqn = 2
y = np.ndarray((nt, neqn), dtype=float)
y0 = np.array([1.0, 0.0])
t0 = 0
tf = 6 * np.pi

yfin_SHO, t_SHO = rk4.rk4(rhs_SHO, y, y0, t0, tf, nt, neqn)

# plot y(t) versus t
plt.figure(1)
plt.plot(t_SHO, yfin_SHO[:,0])

# plot y'(t) versus y(t)
plt.figure(2, figsize=(5,5))
plt.plot(yfin_SHO[:,0], yfin_SHO[:,1])
plt.title("y'(t) versus y(t)")


#------------------Solutions to the Rossler System---------------------

a = 0.2
b = 0.2
c = 4.2

@njit
def rhs_Rossler(t, y):
    """
    Parameters
    ----------
    t : int
        time step at which are evaluating the rhs.
    y : array
        y vector (of length 3, (x, y, z)) at that time, for which each of the 
        3 equations could be dependent.

    Returns
    -------
    f : array
        1D NumPy array holding rhs func eval of each.
    
    """
    
    f = np.zeros(3)
    f[0] = -y[1] - y[2]
    f[1] = y[0] + a*y[1]
    f[2] = b + y[2]*(y[0] - c)
    
    return f

nt = 10000
neqn = 3
y1 = np.ndarray((nt, neqn), dtype=float)
y2 = np.ndarray((nt, neqn), dtype=float)
y01 = np.array([0.0, -6.78, 0.02])
y02 = np.array([0.2, -6.58, 0.22])
t0 = 0
tf = 200

yfin_Rossler, t_Rossler = rk4.rk4(rhs_Rossler, y1, y01, t0, tf, nt, neqn)

# plot x(t) versus t
plt.figure(3, figsize=(10,7))
plt.plot(t_Rossler, yfin_Rossler[:,0])

# plot (x(t), y(t), z(t))
fig = plt.figure(4, figsize=(15,10))
ax1 = fig.add_subplot(111, projection='3d')
ax1.plot(yfin_Rossler[:,0], yfin_Rossler[:,1], yfin_Rossler[:,2])
plt.tight_layout()


# plot z(t) versus t
plt.figure(5, figsize=(10,7))
plt.plot(t_Rossler, yfin_Rossler[:,2])

# plot y(t) versus x(t)
plt.figure(6, figsize=(10,9))
plt.plot(yfin_Rossler[:,0], yfin_Rossler[:,1])

# plot z(t) versus x(t)
plt.figure(7, figsize=(10,7))
plt.plot(yfin_Rossler[:,0], yfin_Rossler[:,2])


    
    
    





