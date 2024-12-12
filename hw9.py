# -*- coding: utf-8 -*-
"""
@author: Annika Deutsch
@date: 11/15/2023
@title: HW 9
@CPU: 12th Gen Intel(R) Core(TM) i7-1255U
@Operating System: Windows 11 Home 64 bit 
@Interpreter and version no.: Python 3.10.9
"""

import random as rng
import numpy as np
import matplotlib.pyplot as plt
from numba import njit

rng.seed(None)

#-------------test the rng with histogram of vals from (0,1)-------------------

N = 100000
bins = np.linspace(0.0, 1.0, 100)
counts = np.zeros(len(bins), dtype=int)
vals = np.zeros(N, dtype=float)

for i in range(len(vals)):
    vals[i] = rng.random()
    binned = True
    j = 0
    while binned:
        if vals[i] < bins[j]:
            counts[j] += 1
            binned = False
        j += 1

plt.figure(1, figsize=(9,6))
plt.plot(bins, counts/N)

#----------------test the rng by plotting pairs of rn's------------------------
M = 10000
rands = [rng.random() for i in range(M)]
pairs = np.zeros((2,M), dtype=float)

for i in range(len(pairs[0])-1):
    pairs[0][i] = rands[i]
    pairs[1][i] = rands[i+1]
    
plt.figure(2, figsize=(9,6))
plt.scatter(pairs[0], pairs[1], alpha=0.5)


#-------------------------Monte-Carlo calculation------------------------------

# subroutines------------------------------------------------------------------

# energy calculation subroutine
@njit
def int_energy(int_matrix, protein):
    """
    Parameters
    ----------
    int_matrix : array
        20 by 20 symmetric array containing the interaction energies between 
        different types of amino acids
        
    protein : array
        (3, N_L) array containing the x position, y position, and type of 
        each amino acid

    Returns
    -------
    E : float
        total energy of the protein in that configuration
    """
    E = 0

    for i in range(len(protein[0])):
        for j in range(i+1, len(protein[0])):
            calc = False
            
            xi = protein[0][i]
            xj = protein[0][j]
            yi = protein[1][i]
            yj = protein[1][j]
            
            if abs(j-i) > 1:
                if abs(xi - xj) + abs(yi - yj) == 1:
                    calc = True
            
            if calc:
                ti = protein[2][i]
                tj = protein[2][j]
                
                E += int_matrix[ti][tj] 
    return E

# amino acid distance subroutine
@njit
def amino_dist(protein, i, j):
    """
    Parameters
    ----------
    protein : array
        (3, N_L) array containing the x position, y position, and type of 
        each amino acid
        
    i : int
        array position of the first amino acid
        
    j : int
        array position of the second amino acid

    Returns
    -------
    xdist : int
        distance on the x axis between the two amino acids
        
    ydist : int
        distance on the y axis between the two amino acids
    """
    xi = protein[0][i]
    xj = protein[0][j]
    yi = protein[1][i]
    yj = protein[1][j]
    
    xdist = abs(xi - xj)
    ydist = abs(yi - yj)
    
    return xdist, ydist

# generate and check validity of protein change subroutine
#@njit
def generate_change(protein):
    """
    Parameters
    ----------      
    protein : array
        (3, N_L) array containing the x position, y position, and type of 
        each amino acid

    Returns
    -------
    protein : array
        modified (3, N_L) array containing the x position, y position, and 
        type of each amino acid. The modification is a shift by +/- 1 in x or 
        y of one amino acid
    """
    still_looking = True
    while still_looking:
        temp_protein = protein.copy()
        a = rng.randrange(start=0, stop=45, step=1)
        direction = rng.randrange(start=1, stop=4, step=1)
        
        if direction == 1:
            temp_protein[0][a] = protein[0][a] + 1 # x+1
            temp_protein[1][a] = protein[1][a] + 1 # y+1
            
        if direction == 2:
            temp_protein[0][a] = protein[0][a] - 1 # x-1
            temp_protein[1][a] = protein[1][a] + 1 # y+1
            
        if direction == 3:
            temp_protein[0][a] = protein[0][a] + 1 # x+1
            temp_protein[1][a] = protein[1][a] - 1 # y-1
            
        if direction == 4:
            temp_protein[0][a] = protein[0][a] - 1 # x-1
            temp_protein[1][a] = protein[1][a] - 1 # y-1
        
        occupied = False
        
        for i in range(len(temp_protein)):
            xdist, ydist = amino_dist(temp_protein, i, a)
            dist = xdist + ydist
            if i != a and dist == 0:
                occupied = True
                break
        
        if occupied == False:
            distup = 1
            distdown = 1
            if a != 44:
                xup, yup = amino_dist(temp_protein, a, a+1)
                distup = xup + yup
            if a != 0:
                xdown, ydown = amino_dist(temp_protein, a, a-1)
                distdown = xdown + ydown
            
            if distup != 1 or distdown != 1:
                occupied = True
                
        if occupied:
            still_looking = True
        else:
            still_looking = False
            break
    
    return temp_protein

# initialize the interaction matrix--------------------------------------------
int_mat = np.zeros((20,20), dtype=float)
Emin = -7
Emax = -2

for i in range(len(int_mat[0])):
    for j in range(i,len(int_mat[0])):
        rn = rng.uniform(Emin, Emax)
        int_mat[i][j] = rn
        int_mat[j][i] = rn
        
#initialize protein structure and calculate initial energy---------------------
N_L = 45
protein = np.zeros((3,N_L), dtype=int) # row 0:x, 1:y, 2:amino acid type

for i in range(len(protein[1])): # starts out as vertical line on x = 22
    protein[0][i] = 22
    protein[1][i] = i
    protein[2][i] = rng.randrange(start=0, stop=19)
    
E0 = int_energy(int_mat, protein)

#step through structure change steps 10**7 times-------------------------------
plt.figure(3, figsize=(9,6))
plt.scatter(protein[0], protein[1], label='Initial Protein')
plt.title('Initial Protein')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0,45)

#set up energy array to fill in
energy = np.zeros(500)

N_step = 10.0**7.0
record = N_step / 500
i = 0 # index to track number of steps of MC calc
j = 0 # index to track energies and distances recorded
while i < N_step:
    new = generate_change(protein)
    Enew = int_energy(int_mat, new) # calculate new interaction energy
    DeltaE = Enew - E0
    
    if DeltaE <= 0:
        protein = new.copy()
        E0 = Enew
        
        if i == 10.0**4.0:
            protein4 = new.copy()
        if i == 10.0**5.0:
            protein5 = new.copy()
        if i == 10.0**6.0:
            protein6 = new.copy()
            
        #record the energies at various intervals
        if i % record == 0:
            energy[j] = E0
            j += 1
        
        i += 1
    else:
        pE = np.exp(-DeltaE)
        R = rng.random()
        if pE > R:
            protein = new.copy()
            E0 = Enew
            
            if i == 10.0**4.0:
                protein4 = new.copy()
            if i == 10.0**5.0:
                protein5 = new.copy()
            if i == 10.0**6.0:
                protein6 = new.copy()
                
            #record the energies at various intervals
            if i % record == 0:
                energy[j] = E0
                j += 1
            
            i += 1
            
plt.scatter(protein[0]+5, protein[1], label='Final Protein')
plt.legend()
            

plt.figure(4, figsize=(9,6))       
plt.scatter(protein4[0], protein4[1], label='Nstep = 10^4')
plt.title('Nstep = 10^4')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0,45)

plt.figure(5, figsize=(9,6))
plt.scatter(protein5[0], protein5[1], label='Nstep = 10^5')
plt.title('Nstep = 10^5')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0,45)

plt.figure(6, figsize=(9,6))
plt.scatter(protein6[0], protein6[1], label='Nstep = 10^6')
plt.title('Nstep = 10^6')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0,45)

plt.figure(7, figsize=(9,6))
plt.scatter(protein[0], protein[1], label='Final Protein')
plt.title('Final Protein')
plt.gca().set_aspect('equal', adjustable='box')
plt.xlim(0,45)

vals = np.linspace(0, N_step, 500)
plt.figure(7, figsize=(9,6))
plt.plot(vals, energy)
plt.title('Protein Energy with Time')
plt.xlabel('Time Step')
plt.ylabel('Energy')

            

            




