#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 4

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

#4.2
def phi(x, y):
    return x**4 - x**2 + y**4 - 0.2 * y**3 - y**2 + 0.2 * x * y**3

def F(x, y):
    return np.matrix([[-4.0*x**3 + 2.0*x - 0.2*y**3]], [[-4.0*y**3 + 0.6*y**2 + 2.0*y - 0.6*x*y**2]])
    
x = np.arange(-1.0, 1.01, 0.1)
y = np.arange(-1.0, 1.01, 0.1)

x, y = np.meshgrid(x, y)
Axes3D(plt.figure()).plot_wireframe(x, y, np.vectorize(phi)(x, y))
plt.show()

