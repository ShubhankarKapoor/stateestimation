#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 08:20:23 2022

@author: shub
"""

import numpy as np
import scipy as sp
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from scipy.sparse import lil_matrix

import matplotlib.pyplot as plt

def func(arr, v0, p0, q0, A, B):

    # print(len(arr))    
    if len(arr)%3 == 0:
        N = int(len(arr)/3)
    # print(N)
    p = arr[:N] # vector of active loads
    q = arr[N: 2*N] # vector of reactive loads
    v = arr[2*N:] # shouldnt this just be v0?
    h = 1e-4

    F1 = np.zeros(N)
    F2 = np.zeros(N)
    F3 = np.zeros(N)

    F1[0] = p[0] - p0 + h*(-1. - (p0**2 + q0**2) / v0**2) # not sure what is this
    F2[0] = q[0] - q0 + h*(A - B*(p0**2 + q0**2) / v0**2) # not sure about this either
    F3[0] = v[0]**2 - v0**2 - 2*h*(p0 + B*q0) # slack voltage
    # F3[0] = v0
    # print(F1[0], F2[0], F3[0])
    for i in range(1, N):
        # print(i)
        F1[i] = p[i] - p[i-1] + h*(-1. - (p[i-1]**2 + q[i-1]**2) / v[i-1]**2) # active power line flows dependent on state vars
        F2[i] = q[i] - q[i-1] + h*(A - B*(p[i-1]**2 + q[i-1]**2) / v[i-1]**2) # reactive power line flows dependent on state vars
        F3[i] = v[i]**2 - v[i-1]**2 - 2*h*(p[i-1] + B*q[i-1]) # voltages dependent on state vars

# backward sweep
#     F1[N-1] = 0 - p[N-1] + h*r[N-1] * (p[N-1]**2 + q[N-1]**2) / v[N-2]**2 + h*P[N-1]  
#     F2[N-1] = 0 - q[N-1] + h*x[N-1] * (p[N-1]**2 + q[N-1]**2) / v[N-2]**2 + h*Q[N-1] 

# forward sweep
#     for i in xrange(1, N):
#         F3[i] = v[i]**2 - v[i-1]**2 + 2*h*(r[i]*p[i] + x[i]*q[i]) #- (r[i]**2 + x[i]**2)*(p[i]**2 + q[i]**2) / v[i-1]**2)
#     F3[0] = v[0]**2 - v0**2 + 2*h*(r[0]*p[0] + x[0]*q[0])  #+ (r[0]**2 + x[0]**2)*(p[0]**2 + q[0]**2) / v0**2) 

    return np.concatenate((F1, F2, F3), axis=0)

def jacob(arr, v0, p0, q0, A, B):

    if len(arr)%3 == 0:
        N = int(len(arr)/3)
    J = lil_matrix((3*N, 3*N))
    p = arr[:N]
    q = arr[N: 2*N]
    v = arr[2*N:]
    h = 1e-4

    J[0,0] = 1.
    J[N,N] = 1.
    J[2*N, 2*N] = 2*v[0]

    for i in range (1,N):
        # jacobian for pflow
        J[i, i-1] = -1. - 2*h*p[i-1] / v[i-1]**2
        J[i, i] = 1.
        J[i, i+N-1] = -2*h*q[i-1] / v[i-1]**2
        J[i, i+2*N-1] = 2*h*(p[i-1]**2 + q[i-1]**2) / v[i-1]**3

        # jacobian for qflow    
        J[i+N, i-1] = -2*h*B*p[i-1] / v[i-1]**2
        J[i+N, i+N] = 1.
        J[i+N, i+N-1] = -1. - 2*h*B*q[i-1] / v[i-1]**2
        J[i+N, i+2*N-1] = 2*h*B*(p[i-1]**2 + q[i-1]**2) / v[i-1]**3

        # jacobian for voltage
        J[i+2*N, i-1] = -2*h # with pline
        J[i+2*N, i+N-1] = -2*h*B # with qline
        J[i+2*N, i+2*N-1] = -2*v[i-1] # with preceeding
        J[i+2*N, i+2*N] = 2*v[i] # with self

    J = csc_matrix(J)
    return J

import scipy as sp
import scipy.optimize

N = 5
v0 = 1.
p0 = 0.
q0 = 0.
A = -0.5
r = 1.
x = 1.
B = r/x
arr0 = np.ones(3*N)

# x = np.ones(N)
# r = np.ones(N)
# P = np.zeros(N)
# Q = np.zeros(N)
# P[:] = 1.
# Q[:] = 0.5

#rhs = func(arr0, x, r, P, Q, v0)
def func_newton(arr): return func(arr, v0, p0, q0, A, B) 
def jacob_newton(arr): return jacob(arr, v0, p0, q0, A, B)
def newton(x, tolF, toldx, maxiters):

    for i in range(maxiters):
        F = func_newton(x)
        J = jacob_newton(x)
        # print(F.shape())
        # print(J.shape())
        dx = sp.sparse.linalg.spsolve(J, -F)
        normF = np.linalg.norm(F)
        normdx = np.linalg.norm(dx)
        x = x + dx
        if normF < tolF and normdx < toldx:         
            print ('Converged to x= {} in {} iterations'.format(x, i) )
            return x, J

        if iter == maxiters:    
            print ('Non-Convergence after {} iterations!!!'.format(i))
            return x, J

sol, J = newton(arr0,1e-3, 1e-3, 50)
#sol = sp.optimize.broyden1(func_newton, arr0, f_tol = 10e-3)

s = np.linspace(0.0001,0.5,N)
z = 0.5 - s
f, (ax1,ax2) = plt.subplots(2, 1)
ax1.plot(z, sol[:N]/sol[3*N-1], 'g')
ax1.plot(z, sol[N:2*N]/sol[3*N-1], 'r')
ax2.plot(z, sol[2*N:]/sol[3*N-1])
#plt.axis([0, 5000, 0.9, 1.1])
#plt.show()
