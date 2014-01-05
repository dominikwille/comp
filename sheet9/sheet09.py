#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 9
#
#Packete einlesen:
import numpy as np
import matplotlib.pyplot as plt
from scipy import special

#Aufgabe 9.1.1:

def next(Fx, Fy, vx, vy, t, h, g, k):
    K1x = Fx(t, vx, vy, k, g)
    K1y = Fy(t, vx, vy, k, g)

    K2x = Fx(t + h/2., vx + K1x * h/2., vy + K1y * h/2., k, g)
    K2y = Fy(t + h/2., vx + K1x * h/2., vy + K1y * h/2., k, g)

    K3x = Fx(t + h/2., vx + K2x * h/2., vy + K2y * h/2., k, g)
    K3y = Fy(t + h/2., vx + K2x * h/2., vy + K2y * h/2., k, g)

    K4x = Fx(t + h, vx + K3x * h, vy + K3y * h, k, g)
    K4y = Fy(t + h, vx + K3x * h, vy + K3y * h, k, g)

    vx = vx +  h * (K1x + 2*K2x + 2*K3x + K4x) / 6
    vy = vy +  h * (K1y + 2*K2y + 2*K3y + K4y) / 6
    t = t + h
    
    return (t, vx, vy)

def Fx(t, vx, vy, k, g):
    return - k * np.sqrt(vx**2 + vy**2) * vx

def Fy(t, vx, vy, k, g):
    return - k * np.sqrt(vx**2 + vy**2) * vy - g

#Aufgabe 9.1.2
#Ermittle alpha mittels Bisektionsverfahren.

x0 = 0
y0 = 0
v0 = 100
k = 0.004
g = 9.81
x_p = 200
v_p = 1

alpha_min = 0
alpha_max = np.pi / 2.

h = 0.0001

# while(alpha_max - alpha_min > 0.00001):
#     alpha = (alpha_min + alpha_max) / 2.
#     vx = np.cos(alpha) * v0
#     vy = np.sin(alpha) * v0
#     x = x0
#     y = y0
#     t = 0
#     p_x = []
#     p_y = []
#     while(x <= x_p):
#         p_x.append(x)
#         p_y.append(y)
#         x += h * vx
#         y += h * vy
#         (t, vx, vy) = next(Fx, Fy, vx, vy, t, h, g, k)

#     if(50 - t * v_p > y):
#         alpha_min = alpha
#     else:
#         alpha_max = alpha

# print 'Alpha: ' + str(round(alpha / np.pi * 360, 2))
# print 'Höhe: '+ str(round(y, 2))
# plt.plot(p_x, p_y)
# plt.show()

#Um den Fallscirmspringer zu treffen muss ein  Winkel von alpha=49.55° gewählt werden. Der Fallschirmspriger wird in einer Höhe von 46.54m getroffen.

def V1(x):
    return 2.5e-4 * x**8

def V2(x):
    return 10e-4*x**8

def F1(x, Phi, phi, V, (E, m, h_bar)):
    return phi

def F2(x, Phi, phi, V, (E, m, h_bar)):
    return 2 * m * (V(x) - E) * Phi / h_bar




h_bar = 1
m = 1
h = 0.01

# E_min = 0
# E_max = 100
# while(E_max - E_min > 0.01):
P1 = []
P2 = []
I_min = 100000
x_min = -6
E_min = 8.00
E_max = 16.00
for i in range(100):
    E = (E_max + E_min) / 2.
    x = x_min
    Phi = 10e-10
    phi = 10e-10
    p_x = []
    p_y = []
    I = 0
    while(x <= - x_min):
        I += phi**2
        (x, Phi, phi) = next(F1, F2, Phi, phi, x, h, (E, m, h_bar), V1)

    if(phi > 0):
        E_min = E
    else:
        E_max = E
    print 'E:' + str(E) + ' I:' + str(I)

x = x_min = -5.0
Phi = 10e-10
phi = 10e-10
p_x = []
p_y = []
while(x <= -x_min):
    p_x.append(x)
    p_y.append(Phi)
    (x, Phi, phi) = next(F1, F2, Phi, phi, x, h, (E, m, h_bar), V1)


plt.plot(p_x, p_y)
plt.show()

#Durch verschiede E_min und E_max wurden die folgenden Energieeigenwerte ermittelt:
#E1 = 0.134
#E2 = 1.120
#E3 = 2.822
#E4 = 5.064
#E5 = 7.767
