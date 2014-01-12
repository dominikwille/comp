#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 10
#
#Packete einlesen:
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

# Aufgabe 10.1.1

def u0(x, U):
    return 0.5 * U * np.exp(-np.abs(x) * U)

def B(n, U):
    return 2 * (1 - np.cos(n*np.pi)*np.exp(-U/2)) / (1 + (2*np.pi*n / U)**2)

def u(x, t, n, U, D=1):
    val = B(0, U) / 2
    for n in range(1, n+1):
        val += D * np.exp(-t * (2 * np.pi * n)**2) * B(n, U) * np.cos(2 * n * np.pi * x)
    return val

x = np.arange(-0.5, 0.5, 0.01)

# plt.plot(x, u(x, 0.05, 100, 10))
# plt.plot(x, u0(x, 10)) #Um einen vergleich zu haben.
# plt.show()

#Aufgabe 10.1.2

def next1(u, dx, dt):
    N = len(u)
    i = 0
    v = []
    while(i < N):
        if(i == 0):
            v.append(u[i] + dt/dx**2 * (u[i] - 2*u[i] + u[i+1]))
        elif(i == N - 1):
            v.append(u[i] + dt/dx**2 * (u[i-1] - 2*u[i] + u[i]))
        else:
            v.append(u[i] + dt/dx**2 * (u[i-1] - 2*u[i] + u[i+1]))
        i += 1
    return v

#Aufgabe 10.1.4
def next2(u, dx, dt):
    N = len(u)
    u = np.matrix(u) / dt
    A = np.matrix([[0] * N] * N)
    i = 0
    while(i < N):
        A[i,i] = 1/dt + 2/dx**2
        if(i > 0):
            A[i,i-1] = -1/dx**2
        if(i < N - 1):
            A[i,i+1] = -1/dx**2
        if(i == 0 or i == N - 1):
            A[i,i] = 1/dt + 1/dx**2
        i += 1
    return np.linalg.solve(A, u.T).A1

#Aufgabe 10.1.7
#Zu sehen ist, dass bei zu großem dt die Funktion eine spitze in der mitte behält!
def next3(u, dx, dt):
    N = len(u)
    A = np.matrix([[0] * N] * N)
    i = 0
    while(i < N):
        A[i,i] = 1/dt - 1/dx**2
        if(i > 0):
            A[i,i-1] = 1/(2*dx**2)
        if(i < N - 1):
            A[i,i+1] = 1/(2*dx**2)
        if(i == 0 or i == N - 1):
            A[i,i] = 1/dt - 1/(2*dx**2)
        i += 1
    u = A * np.matrix(u).T

    A = np.matrix([[0] * N] * N)
    i = 0
    while(i < N):
        A[i,i] = 1/dt + 1/dx**2
        if(i > 0):
            A[i,i-1] = -1/(2*dx**2)
        if(i < N - 1):
            A[i,i+1] = -1/(2*dx**2)
        if(i == 0 or i == N - 1):
            A[i,i] = 1/dt + 1/(2*dx**2)
        i += 1
    return np.linalg.solve(A, u).A1

#Aufgabe 10.1.8
def next4(u, dx, dt):
    K = 10
    Beta = 5000
    N = len(u)
    i = 0
    v = []
    while(i < N):
        x = -0.5 + i * dx
        if(i == 0):
            v.append(u[i] + dt/dx**2 * (u[i] - 2*u[i] + u[i+1]) - dt*Beta*u[i]*np.exp(-abs(x-0.5)*K))
        elif(i == N - 1):
            v.append(u[i] + dt/dx**2 * (u[i-1] - 2*u[i] + u[i]) - dt*Beta*u[i]*np.exp(-abs(x-0.5)*K))
        else:
            v.append(u[i] + dt/dx**2 * (u[i-1] - 2*u[i] + u[i+1]) - dt*Beta*u[i]*np.exp(-abs(x-0.5)*K))
        i += 1
    return v

def concentration(ts, U, dx, dt):
    c = []
    for t in ts:
        print t
        c.append(sum(iterate(t, u0(x, U), dx, dt, next4)) * dx)
    return c

def position(ts, U, dx, dt, x):
    p = []
    a = iterate(0.1, u0(x, U), dx, dt, next4)
    print np.matrix(a)
    print np.matrix(a) * np.matrix(x).T * dx
    for t in ts:
        print t
        z = np.matrix(iterate(t, u0(x, U), dx, dt, next4)) * np.matrix(x).T * dx
        p.append(z[0,0])
    return p

#Hilfsfunktion um zu einer bestimmten zeit zu iterieren
def iterate(t_max, u, dx, dt, next):
    t = 0
    while(t <= t_max):
        u = next(u, dx, dt)
        t += dt
    return u

dt = 0.00001
dx = 0.01
x = np.arange(-0.5, 0.5, dx)
t = np.arange(0.0, 0.2, 0.005)
U = 10
t_max = 0.2
# u1 = iterate(t_max, u0(x, U), dx, dt, next1) 
# u2 = iterate(t_max, u0(x, U), dx, dt, next2)
# u3 = iterate(t_max, u0(x, U), dx, dt, next3)
# u4 = iterate(t_max, u0(x, U), dx, dt, next4)
# c = concentration(t, U, dx, dt)
# p = position(t, U, dx, dt, x)
# print c
#Aufgabe 10.1.3/5
r = dt/dx**2 
print r
#10.1.3: Es sollte r <= 0.1 gelten um eine gute Konvergenz zu erreichen.
#10.1.5: Mit dem Rückwertszeitschritt ist das Ergebnis viel stabiler!
#10.1.6: Beim Vorwärtseinsetzen werden für jede Stelle nur grundrechearten angewendet,
#        ein Iterationsschritt ist daher nicht sehr Aufwändig. Beim Rückwertseinsetzen
#        muss bei jedem iterationsschritt ein lineares Gleichungsystem gelöst werden.
#        Da die Matrix aber fast schon Diagonalform hat ist das auch nicht so Aufwendig.       


# plt.plot(x, u0(x, U)) #Auch hier wieder zum vergleichen
# plt.plot(x, u(x, t_max, 100, U))
# plt.plot(x, u1)
# plt.plot(x, u2)
# plt.plot(x, u3)
# plt.plot(x, u4)
# plt.plot(t, c)
# plt.plot(t, p)
# plt.show()
