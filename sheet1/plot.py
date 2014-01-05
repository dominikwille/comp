#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @autor Dominik Wille
# @tutor Alexander Schlaich
# @exercise 1.3
#
# 
# Import math-Modulte to compare my tcos function to an other one.
import math
import matplotlib
import numpy
from pylab import *

def tcos(x, N):
    cos = 0;
    for i in range(0, N):
        cos += ((-1)**i)*x**(2*i)/math.factorial(2*i)
    return cos

print 'the cos of 3.142 is ' + str(tcos(3.142, 50))

t = arange(-1.0, 1.0, 0.01)
s = tcos(2*pi*t, 7)
plot(t, s)

xlabel('x')
ylabel('tcos(x, N)')
grid(True)
savefig("test.png")
show()
