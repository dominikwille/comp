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

def u0 (x, U):
    return 0.5 * U * np.exp(-np.abs(x) * U)

def B (n, U):
    return 2 * (1 - np.cos(n*np.pi)*np.exp(-U/2)) / (1 + (2*np.pi*n / U)**2)

def u (x, t, n, U, D=1):
    val = B(0, U) / 2
    for n in range(1, n+1):
        val += D * np.exp(-t * (2 * np.pi * n)**2) * B(n, U) * np.cos(2 * n * np.pi * x)
    return val

x = np.arange(-0.5, 0.5, 0.001)

plt.plot(x, u(x, 0, 10, 10))
plt.plot(x, u0(x, 10))
plt.show()
