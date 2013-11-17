#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @autor Dominik Wille
# @tutor Alexander Schlaich
# @exercise 1.3
#
# 
# Import math-Modulte to compare my tcos function to an other one.
import math as m
import matplotlib as plt
import numpy as np
from pylab import *

def tcos(x, N):
    cos = 0;
    for i in range(0, N):
        cos += ((-1)**i)*x**(2*i)/math.factorial(2*i)
    return cos

def fak(arr):
    response = []
    for n in arr:
        result = 1
        while(n > 1):
            result *= n
            n -= 1
        response.append(result)
    return response

# print 'the cos of 3.142 is ' + str(tcos(3.142, 50))

t = np.arange(1, 10, 1)

# t = [1, 2, 3]
# s = tcos(2*pi*t, 7)
plt.plot(t, fak(t))

plt.grid(True)
plt.show()
