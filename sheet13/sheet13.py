#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 13
#
#Packete einlesen:
import numpy as np
import os
import csv
import matplotlib.pyplot as plt
import random as rnd

#Aufgabe 13.1.1
#Für das integral von 0 bis delta ist x << 1 und daher
#kann 1 + x nach taylor als e**x genähert werden.
#Das Inegral ist somit 2*sqrt(delta).
def f(x):
    return 1.0/np.sqrt(np.log(1+x))

def hit_or_miss(f, a, b, N, f_min = None, f_max = None):
    if(f_min is None):
        f_min = f(b)
    if(f_max is None):
        f_max = f(a)
    N_pos = 0
    for i in range(N): 
        r = rnd.random()
        s = rnd.random()
        if(f(a + (b - a) * r) - f_min > (f_max - f_min) * s):
            N_pos += 1

    return (b - a) * (float(N_pos) / float(N) * (f_max - f_min) + f_min)


# delta = 10e-3
# a = delta
# b = 1
# N = 1000
# I = []
# for i in range(10):
#     I.append(hit_or_miss(f, a, b, N) + 2*np.sqrt(delta))

# print np.average(I)
# print np.std(I)
#Durch 100 mal mehr schritte wird die Standardabweichung
#ca 10 mal kleiner.

#Aufgabe 13.1.2
#Der grenzwert für x->0 ist für f(x)/g(x) ist 1,
#weil man 1 + x wieder als e**x nähern kann.

def f2(y):
    return 1.0/np.sqrt(np.log(1+y*2/4.0))*y/2
    
a = 0
b = 2
N = 100000
I = []
# for i in range(10):
#     I.append(hit_or_miss(f2, a, b, N, f_max = 1))

# print np.average(I)
# print np.std(I)
#Die Standardabweichung ist um midestens eine Größenordnung
#(10 mal) kleiner als in 13.1.1. Bemerkswert ist auch,
#dass das Integral einen merklich anderen wert annimmt.
#evtl fehlerhafte implementierung???

#Aufgabe 13.2

def rand_m(N):
    X = np.matrix([[0]*N]*N)
    for i in range(N):
        for j in range(N):
            X[i,j] = rnd.choice([-3,-1,1,3])
    return X

def repeat(x, N):
    while x < 0:
        x += N
    while x >= N:
        x -= N
    return x

def flip(x):
    if(x == -3):
        return -1
    elif(x == 3):
        return 1
    else:
        return x + rnd.choice([-2, 2])

def getdE(X, (y1, y2), n_element, H, J):
    dE = 0

    dE += - H * n_element / 2.0
    dE -= - H * X[y1, y2] / 2.0

    if(y1 > 0):
        dE += - J * X[y1 - 1, y2] * n_element / 8.0 
        dE -= - J * X[y1 - 1, y2] * X[y1, y2] / 8.0
    if(y1 < N - 1):
        dE += - J * X[y1 + 1, y2] * n_element / 8.0
        dE -= - J * X[y1 + 1, y2] * X[y1, y2] / 8.0
    if(y2 > 0):
        dE += - J * X[y1, y2 - 1] * n_element / 8.0 
        dE -= - J * X[y1, y2 - 1] * X[y1, y2] / 8.0
    if(y2 < N - 1):
        dE += - J * X[y1, y2 + 1] * n_element / 8.0
        dE -= - J * X[y1, y2 + 1] * X[y1, y2] / 8.0

def metropolis((x1, x2), X, kT, H, J=1.0, r=1.0, steps=10):
    N = len(X)
    for i in range(steps):
        (q1, q2) = (int(round(rnd.random() * 2 * r - r)), int(round(rnd.random() * 2 * r - r)))
        (y1, y2) = (repeat(x1 + q1, N), repeat(x2 + q2, N))
        n_element = flip(X[y1,y2])
        dE = getdE(X, (y1, y2), n_element, H, J)
        if(dE < 0):
            X[y1, y2] = n_element
            (x1, x2) = (y1, y2)
        else:
            p_A = min(1.0, np.exp(-dE/kT))
            if(rnd.random() < p_A):
                X[y1, y2] = n_element
                (x1, x2) = (y1, y2)
        
    return X

N = 10
X = rand_m(N)
print metropolis((4, 4), X, 10e-3, 0, steps=25000)

plt.imshow(X, extent=(0, N, N, 0), interpolation='nearest', cmap=plt.cm.jet)
plt.show()
