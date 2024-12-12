# -*- coding: utf-8 -*-
"""
@author: Annika Deutsch
@date: 10/28/2023
@title: HW 7
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

# imports
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

# define constants
hb = 6.5821e-16 #eV-sec
kepf = 3.801 #eV-Angstroms^2, Kinetic Energy Pre-Factor
L = 500.0 #Angstroms, region to perform calculation
s = 10.0 #Angstroms, width
k0 = 1.0 #Angstrom^-1, avg wavenumber
v1 = 2.0 #eV
x0 = 0.5 * L #Angstroms
omegax = 5.0 #angstroms

# define function that describes potential:
def V(x):
    if x < x0:
        return v1 * (0.75 - np.cos((x - x0)/omegax))
    else:
        return 0 
    
# define function for initial wf:
def psi0(x):
    return np.exp(-((x-(0.3*L))/s)**2 + (x*k0*1j))

# define function for mod squared of psi
def psisq(x):
    return np.abs(np.exp(-((x-(0.3*L))/s)**2 + (x*k0*1j)))**2

x = np.linspace(0, L, 1000)
vx = np.zeros(len(x))
psi0x = np.empty(len(x), dtype= complex)
for i in range(len(x)):
    vx[i] = V(x[i])
    psi0x[i] = psi0(x[i])
    
psiint, err = integrate.quad(psisq, -np.inf, np.inf)
print(psiint)
    
plt.figure(1)
plt.plot(x, vx)

plt.figure(2)
plt.plot(x, np.imag(psi0x))

plt.figure(3)
plt.plot(x, np.real(psi0x))

plt.figure(4)
plt.plot(x, np.abs(psi0x))

    
