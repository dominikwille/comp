#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 7
#
#Packete einlesen:
import numpy as np

#7.1.2
def D(func,x,h):
	return (func(x+h)-func(x))/h


def D2(func,x,h):
	return (func(x+h)-func(x-h))/(2*h)

#7.1.3
def f(x):
	return 1/(2+np.cos(x))
	
x = np.array[0,2*np.pi]
print x
