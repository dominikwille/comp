#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 11
#
#Packete einlesen:
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import special

# Aufgabe 11.1.1
P_ = []
dt = 0.1
T = 0
N = 1
with open('P.txt', 'rb') as csvfile:
     File = csv.reader(csvfile, delimiter=' ')
     for row in File:
          if(row[0] != '#'):
               T += dt
               P_.append(map(float,row))

P = np.matrix(P_)

def Phi0(tau):
     t = 0
     phi = 0
     i = 0
     while(t <= T - tau):
          while(i < N):
               phi += (P[int(t/dt),i] * P[int((t+tau)/dt),i]) * dt
               i += 1
          t += dt
     return phi

print T

# print Phi0(0)
# print Phi0(10)

def id(x):
     return abs(x)**2

# Aufgabe 11.1.2
def Phi1(tau):
     f = np.matrix(P_ + int(T/dt) * [[0,0,0]])
     i = 0
     phi = 0
     while(i < N):
          phi = np.fft.ifft(map(id, np.fft.fft(f[:,i].A1).tolist()))
          # phi = np.fft.ifft(np.fft.fft(f[:,i].A1))
          i += 1
     # return phi.real
     return np.array(map(abs, phi.tolist()))


a = Phi1(0)
a = (a * Phi0(1) / a[10])
a.tolist()

print a[0]
print Phi0(0)
# print a[1]
t = 0
tau = 10
a2 = []
a1 = []
while(t <= tau):
     # a1 = a[int(tau/dt)]
     # a = Phi0(tau)
     if(True):
          a1.append(a[int(t/dt)] / (T - t))
          a2.append(Phi0(t))
     else:
          print 'Error'
     t += dt
     
x = np.arange(0,tau+dt,dt)
# print len(a2)
x = x[0:len(a1)]
plt.plot(x, a2)
plt.plot(x, a1)
# plt.plot(x, P_[:len(x)])
plt.show()

a = [(1 + 1j), (2 + 2j)]

a = map(abs, a)

print a
