# -*- coding: utf-8 -*-
"""
@author: Annika Deutsch
@date: 11/12/2023
@title: HW 8
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import solve, inv
from astropy.table import Table

#--------- write the general purpose lin least sqs functions ----------------

def model(constants, funcs, t):
    vals = np.zeros(len(t), dtype=float)
    for j in range(len(t)):
        sum3 = 0
        for i in range(len(funcs)):
            sum3 += constants[i] * funcs[i](t[j])
        vals[j] = sum3
    return vals

def red_chisq(N, m, data, errs, vals):
    sumchi = 0
    for i in range(len(data)):
        sumchi += ((data[i] - vals[i])/(errs[i]))**2
    
    return (1/(N - m)) * sumchi
    
def linleastsqus(data, variable, errors, basis_funcs):
    Flk = np.zeros((len(basis_funcs), len(basis_funcs)), dtype=float)
    bl = np.zeros(len(basis_funcs), dtype=float)
    a_errs = np.zeros(len(basis_funcs), dtype=float)
    
    for l in range(len(basis_funcs)):
        sum2 = 0
        for i in range(len(variable)):
            sum2 += (1/(errors[i])**2) * basis_funcs[l](variable[i]) * data[i]
        bl[l] = sum2
        for k in range(len(basis_funcs)):
            sum1 = 0
            for j in range(len(variable)):
                sum1 += ((1/(errors[j])**2) * basis_funcs[k](variable[j]) *
                         basis_funcs[l](variable[j]))
            Flk[l][k] = sum1
    
    # use Gauss-Jordan elimination to solve for the prefactors of the basis
    # functions and their errors
    a = solve(Flk, bl)
    Finv = inv(Flk)
    
    # get the model curve
    vals = model(a, basis_funcs, variable)
    
    # get the errors on the model params
    for m in range(len(a_errs)):
        a_errs[m] = np.sqrt(Finv[m][m])
    
    return vals, a, a_errs

#--------------------------- fitting to co2 data -----------------------------

# load in data
data = np.loadtxt('co2_brw_surface-flask_1_ccgg_month.txt', 
                  skiprows=53, usecols=(1,2,3), dtype='float')
t = data[:,0] + data[:,1]/12 # year + month
t_shifted = t - 1971.0 # t to fit the quadratic in
co2 = data[:,2]
errors = [0.004 * y for y in co2]

# plot data vs. t to check
plt.figure(1, figsize=(9,6)) 
plt.plot(t, co2, 'k-')
plt.xlabel('time (year)')
plt.ylabel('CO2 (in ppm)')
plt.title ('original data')
plt.show()

# define the fucntions used in the model
def f1(t):
    return np.sin((2*np.pi)*t)

def f2(t):
    return np.cos((2*np.pi)*t)

def f3(t):
    return np.sin((4*np.pi)*t)

def f4(t):
    return np.cos((4*np.pi)*t)

def f5(t):
    return t**2
    
def f6(t):
    return t
    
def f7(t):
    return 1

# basis function list
fk = [f1, f2, f3, f4, f5, f6, f7]


model_curve, param_vals, param_errs = linleastsqus(co2, t_shifted, errors, fk)
chisq = red_chisq(len(co2), len(fk), co2, errors, model_curve)
print('Full model Chi-Squared: ' + str(chisq))
plt.figure(2, figsize=(9,6)) # plot model_curve vs. t
plt.plot(t, model_curve, 'b-', label='model')
plt.plot(t[519:], co2[519:], 'k-', label = 'data')
plt.xlabel('time (year)')
plt.ylabel('CO2 (in ppm)')
plt.title ('model vs. data')
plt.legend()
plt.show()

params = Table([param_vals, param_errs], names=('Best-Fit Values', 'Errors'))
print(params)
print()

# plot residuals
plt.figure(3, figsize=(9,6)) 
plt.plot(t, model_curve-co2, 'b.')
plt.xlabel('time (year)')
plt.ylabel('model-data (in ppm)')
plt.title ('Residuals')
plt.show()

# without seasonal variations
fk_nos = [f5, f6, f7]
model_curve_nos, params_nos, param_errs_nos = linleastsqus(co2, t_shifted, 
                                                           errors, fk_nos)
plt.figure(4, figsize=(9,6))
plt.plot(t, model_curve_nos, 'b-', label='model without seasonal variation')
plt.plot(t, co2, 'k-', label = 'data')
plt.xlabel('time (year)')
plt.ylabel('CO2 (in ppm)')
plt.title ('no seasons model vs. data')
plt.legend()
plt.show()

# without first harmonic
fk_5 = [f1, f2, f5, f6, f7]
model_curve_5, param_vals_5, param_errs_5 = linleastsqus(co2, t_shifted, 
                                                         errors, fk_5)
chisq_5 = red_chisq(len(co2), len(fk_5), co2, errors, model_curve_5)
print('Without First Harmonic Chi-Squared: ' + str(chisq_5))
plt.figure(5, figsize=(9,6)) # plot model_curve vs. t
plt.plot(t, model_curve_5, 'b-', label='model without first harmonic')
plt.plot(t[419:], co2[419:], 'k-', label = 'data')
plt.xlabel('time (year)')
plt.ylabel('CO2 (in ppm)')
plt.title ('model without first harmonic vs. data')
plt.legend()
plt.show()

params_5 = Table([param_vals_5, param_errs_5], names=('Best-Fit Values', 
                                                      'Errors'))
print(params_5)


