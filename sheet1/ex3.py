#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @autor Dominik Wille
# @tutor Alexander Schlaich
# @exercise 1.3
#
# 
# Import math-Modulte to compare my tcos function to an other one.
import math as math
import matplotlib as plt
import numpy as numpy
import time as time
from pylab import *

def tcos(x, N):
    cos = 0;
    for i in range(0, N):
        cos += ((-1)**i)*x**(2*i)/math.factorial(2*i)
    return cos


x0 = time.clock()

for i in range(1, 100):
    x = tcos(1, i)
    x = time.clock()
    y = numpy.cos(1)
    y = time.clock()
    print 'tcos :' + str(x - x0) + '  numpy: ' + str(y - x)
    x0 = y

t = numpy.arange(0.0, 1.1, 0.01)
plt.plot(t, tcos(2*pi*t, 10))
plt.plot(t, tcos(2*pi*t, 9))
plt.plot(t, numpy.cos(t*2*pi))

plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

print '\nExercise 1.3.2:\n'
print '    As you can see in the plot, the 10th order gives a good approximation of the cos-function.'

print '\nExercise 1.3.3:\n'
print '    My method does not determine any significant difference betwwen the two implementations.'
print '    I would expect a better performance of the numpy implementation.'
