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

def f1(x, y):
    return -4.0*x**3 + 2.0*x - 0.2*y**3

def f2(x, y):
    return -4.0*y**3 + 0.6*y**2 + 2.0*y - 0.6*x*y**2
    
def f1x(x, y):
    return -12.0*x**2 + 2.0

def f1y(x, y):
    return -0.6*y**2

def f2x(x, y):
    return -0.6*y**2

def f2y(x, y):
    return -12.0*y**2 + 1.2*y + 2.0 + 1.2*x*y

def extremum(x, y, n = 100):
    for i in range(0, n):
        a = np.array([[f1x(x, y), f1y(x, y)], [f2x(x, y), f2y(x, y)]])
        b = np.array([-f1(x, y), -f2(x, y)])

        z = np.linalg.solve(a, b)
        x += z[0]
        y += z[1]
    return [x, y]

def pos_def(x):
    for i in np.linalg.eigvals(x):
        if i <= 0:
            return False
    return True

def neg_def(x):
    for i in np.linalg.eigvals(x):
        if i >= 0:
            return False
    return True

#4.2.1
X = np.arange(-1.0, 1.01, 0.1)
Y = np.arange(-1.0, 1.01, 0.1)

# X, Y = np.meshgrid(X, Y)
# Axes3D(plt.figure()).plot_wireframe(X, Y, np.vectorize(phi)(X, Y))
# plt.show()

#4.2.2
#setze Startwerte
x = 0.05
y = 0.05

print extremum(x, y)

#4.2.3
points = []
minima = []
minimum = 0
for x in X:
    for y in Y:
        z = extremum(x, y)
        x_max = z[0]
        y_max = z[1]
        H = np.matrix([[f1x(x_max, y_max), f1y(x_max, y_max)], [f2x(x_max, y_max), f2y(x_max, y_max)]])
        if(neg_def(H)):
            points.append((x, y , 5))
        elif(pos_def(H)):
            if((x_max, y_max) not in minima):
                minima.append((x_max, y_max))
                minimum += 1
            points.append((x, y, minimum))
        else:
            points.append((x, y, 0))

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# for(x, y, z) in points:
#     ax.scatter(x, y, z)
# plt.show()

#4.2.4
