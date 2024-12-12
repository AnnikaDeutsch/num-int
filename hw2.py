"""
@author: Annika Deutsch
@date: 9/6/23
@title: HW 2

@interpreter and version: Python 3.10.9
"""

# import needed packages 
import numpy as np
import math
import matplotlib.pyplot as plt


# define the function to be integrated 
def integrand(x):
    return (x**4.0 * math.e**x) / ((math.e**x - 1.0)**2.0)

#
# use n=1024 (5 sig digs accuracy) to calculate D from 0.01 to 10 with ~200 points bw
#

# define a function that performs the trapezoid rule (algorithm from above)
def trap(xf):
    n = 4 # num intervals
    xa = 0.0
    xb = xf
    dx = (xb-xa)/(n)

    s = 0.5 * (10.0**-10.0 + integrand(xb)) # replace f(0) with value at x = 10**-10

    # add up all the func evals between xa and xb
    for i in range(1, n):
        x = xa + i*dx
        s = s + integrand(x)

    # loop through calculating the sum, doubling n until it reaches 1024
    while n < 1024:
        n = 2*n
        dx = (xb-xa)/n
        # add up all the func evals between xa and xb that have not yet been evaluated
        for i in range(int(n/2)):
            odd = 2*i + 1
            x = xa + odd*dx
            s = s + integrand(x)
        Inew1 = s * dx
    return Inew1

   
#
# use n=1024 (5 sig digs accuracy) to calculate D from 0.01 to 10 with ~200 points bw
#

# define a function that performs simpson's rule, as above
def simp1(xf):
    n = 2 # num intervals (num points = 3)
    xa = 0.0
    xb = xf
    dx = (xb-xa)/(n)

    s1 = 10.0**-10.0 # replace f(0) with value at x = 10**-10
    s2 = 0
    s4 = integrand(0.5*(xa+xb))

    Inew = dx*(s1 + 4*s4)/3

    while n < 1024:
        n = 2*n
        dx = (xb-xa)/n
        s2 = s2 + s4 
        s4 = 0 # reset s4 to recalculate, will be populated by new values
        for i in range(int(n/2)):
            odd = 2*i + 1
            x = xa + odd*dx
            s4 = s4 + integrand(x)
        Inew = dx*(s1 + 2*s2 + 4*s4)/3
    return Inew


# define a function that implements simpson's rule but recalculating s2 and s4 each time
def simp2(ints, x0, xN):
    n = ints # num intervals (num points = 3)
    xa = x0
    xb = xN
    dx = (xb-xa)/(n)

    # endpoint functions have coefficient 1
    s1 = 10.0**-10.0 + integrand(xb)# replace f(0) with 10**-10
    
    # functions of even i (except f0) have coefficient 2 
    s2 = 0 
    for i in range(2, n, 2):
        x = xa + i * dx
        s2 = s2 + integrand(x)
        
    # functions of odd i have coefficient 4
    s4 = 0
    for i in range(1, n, 2):
        x = xa + i * dx
        s4 = s4 + integrand(x)

    Inew2 = dx*(s1 + 2*s2 + 4*s4)/3
    return Inew2

I_simp1 = []
I_simp100 = []
intervals = [2**i for i in range(int(math.log2(4096)) + 1) if 2**i >= 4 and 2**i <= 4096]
for i in intervals:
    I_simp1.append(simp2(i, 0.0, 1.0))
    I_simp100.append(simp2(i, 0.0, 100.0))

    
yvals= np.linspace(0.01, 10, 200)
Dvals_simp = []
Dvals_trap = []
for i in yvals:
    Dvals_simp.append(simp1(i))
    Dvals_trap.append(trap(i))

plt.figure(1)
plt.plot(yvals, Dvals_trap, 'r-', label='Trapezoid Rule')
plt.plot(yvals, Dvals_simp, 'b--', label="Simpson's Rule")
plt.legend()

