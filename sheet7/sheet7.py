#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 6
#
#Packete einlesen:
import numpy as np
import matplotlib.pyplot as plt

#7.1.2
#Definition en der ABleitungsfunktionen und der analytischen Lösung Dana.
def D(func,x,h):
	return (func(x+h)-func(x))/h


def D2(func,x,h):
	return (func(x+h)-func(x-h))/(2*h)
	
def Dana(x):
	return np.sin(x)/(2+np.cos(x))**2

#7.1.3
#Testfunktion und x,h-Werte:
def f(x):
	return 1/(2+np.cos(x))
x = np.arange(0,2*np.pi,0.1)
def h(i):
	return 2**(-i)

#Plot (wurde als figure_1.png exportiert.):
plt.plot(x,Dana(x),'.-',x,D(f,x,h(0)),x,D(f,x,h(1)),x,D(f,x,h(2)),x,D(f,x,h(3)),x,D(f,x,h(4)),x,D(f,x,h(5)),x,D2(f,x,h(0)),'--',x,D2(f,x,h(1)),'--',x,D2(f,x,h(2)),'--',x,D2(f,x,h(3)),'--',x,D2(f,x,h(4)),'--',x,D2(f,x,h(5)),'--')
plt.xlabel('x')
plt.ylabel('D(f(x))')
plt.legend(('analytisch','D1(i=0)','D1(i=1)','D1(i=2)','D1(i=3)','D1(i=4)','D1(i=5)','D2(i=0)','D2(i=1)','D2(i=2)','D3(i=3)','D2(i=4)','D2(i	=5)'), loc=1)
#plt.show()