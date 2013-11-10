#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @autor Dominik Wille
# @tutor Alexander Schlaich
# @sheet 3

import numpy as np
import math as math

def mround(x, N):
    if(x == 0):
        return x
    return round(x, int(N - math.ceil(math.log10(abs(x)))))

def mrnd(x, N):
    f = np.vectorize(mround)
    return f(x, N)

F_ = np.matrix( [ [1.0, -2.0, -1.0], [36000.0, 2.0, 0.0], [-2.0, 1400.0, 1.0] ] )
d_ = np.matrix([[3.0], [72002.0], [1399.0]])

dim = len(d_)

# Exercise 3.1.1
F = np.copy(F_)
d = np.copy(d_)
k = 0
while(k < dim):
    m = F[k,k]
    F[k] /= m
    d[k] /= m
    i = k + 1
    while(i < dim):
        m = F[i,k]
        F[i] -= m * F[k]
        d[i] -= m * d[k]
        i += 1
    k += 1

#erzeuge die einheitsmatrix um den Lösungsvektor zu erhalten.
k = 0
while(k < dim - 1):
    i = k + 1
    while(i < dim):
        f = F[i] * F[k,i] / F[i,i]
        d[k] -= d[i] * F[k,i] / F[i,i]
        F[k] -= f
        i += 1
    k += 1

print 'Exercise 3.1.1: '
print d

#Exercise 3.1.2
F = np.copy(F_)
d = np.copy(d_)
k = 0
while(k < dim):
    m = F[k,k]
    F[k] /= m
    F[k] = mrnd(F[k], 4)
    d[k] /= m
    d[k] = mround(d[k], 4)
    i = k + 1
    while(i < dim):
        m = F[i,k]
        F[i] -= m * F[k]
        F[i] = mrnd(F[i], 4)
        d[i] -= m * d[k]
        d[i] = mround(d[i], 4)
        i += 1
    k += 1

#erzeuge die einheitsmatrix um den Lösungsvektor zu erhalten.
k = 0
while(k < dim - 1):
    i = k + 1
    while(i < dim):
        f = F[i] * F[k,i] / F[i,i]
        d[k] -= d[i] * F[k,i] / F[i,i]
        F[k] -= f
        i += 1
    k += 1
print '\n\n Exercise 3.1.2: '
print d

#Exercise 3.1.3

F = np.copy(F_)
d = np.copy(d_)
k = 0
while(k < dim):

    #nehme Pivotisierung vor
    i = k
    while(i < dim - k - 1):
        if(abs(F[i, k]) < abs(F[i+1, k])):
            t = np.copy(d[i])
            s = np.copy(F[i])
            F[i] = F[i+1]
            d[i] = d[i+1]
            F[i+1] = s
            d[i+1] = t
            i = k
            continue
        i += 1
        
    m = F[k,k]
    F[k] /= m
    F[k] = mrnd(F[k], 4)
    d[k] /= m
    d[k] = mround(d[k], 4)
    i = k + 1
    while(i < dim):
        m = F[i,k]
        F[i] -= m * F[k]
        F[i] = mrnd(F[i], 4)
        d[i] -= m * d[k]
        d[i] = mround(d[i], 4)
        i += 1
    k += 1

#erzeuge die einheitsmatrix um den Lösungsvektor zu erhalten.
k = 0
while(k < dim - 1):
    i = k + 1
    while(i < dim):
        f = F[i] * F[k,i] / F[i,i]
        d[k] -= d[i] * F[k,i] / F[i,i]
        F[k] -= f
        i += 1
    k += 1

print '\n\n Exercise 3.1.3: '
print d
print 'Wie zu sehen ist hilft die pivotisierung dabei den Fehler gering zu halten.'

#Exercise 3.2:

A_ = np.matrix([[64.0, -40.0, 16.0], [-40.0, 29.0, -4.0], [16.0, -4.0, 62.0]])
E_ = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
dim = 3

F = np.copy(A_)
d = np.copy(E_)
k = 0
while(k < dim):
    m = F[k,k]
    i = k + 1
    while(i < dim):
        n = F[i,k] / m
        F[i] -= F[k] * n
        d[i,k] += n
        i += 1
    k += 1

print '\nExercise 3.2.1: '
print '\nL = '
print d
print '\nR = '
print F
print '\nL * R = '
print np.mat(d) * np.mat(F)
