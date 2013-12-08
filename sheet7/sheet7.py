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
# plt.plot(x,Dana(x),'.-',x,D(f,x,h(0)),x,D(f,x,h(1)),x,D(f,x,h(2)),x,D(f,x,h(3)),x,D(f,x,h(4)),x,D(f,x,h(5)),x,D2(f,x,h(0)),'--',x,D2(f,x,h(1)),'--',x,D2(f,x,h(2)),'--',x,D2(f,x,h(3)),'--',x,D2(f,x,h(4)),'--',x,D2(f,x,h(5)),'--')
# plt.xlabel('x')
# plt.ylabel('D(f(x))')
# plt.legend(('analytisch','D1(i=0)','D1(i=1)','D1(i=2)','D1(i=3)','D1(i=4)','D1(i=5)','D2(i=0)','D2(i=1)','D2(i=2)','D3(i=3)','D2(i=4)','D2(i	=5)'), loc=1)
#plt.show()

#7.1.4
def h_extrapolation(func, D_func, h, i, k, x):
        if(k == 0):
                return D_func(func, x, h/(2**i))
        else:
                return h_extrapolation(func, D_func, h, i + 1, k - 1, x) + (h_extrapolation(func, D_func, h, i + 1, k - 1, x) - h_extrapolation(func, D_func, h, i, k - 1, x)) / (2**k - 1)

h = 1.0

# plt.plot(x, Dana(x))
# for i in range(7):
#         plt.plot(x, h_extrapolation(f, D, h, 0, i, x)) 
# plt.show()

def error(func1, func2, values):
        err = 0.0
        for i in values:
                err += abs(func1(i) - func2(i))
        return err

#1. verfahren
l = []
for i in range(0,6):
        print error(Dana, D(func=f, h=h(i)), x)

print l


A = np.array([70, 88, 78, 93, 99, 58, 89, 66, 77, 78])
N = len(A)

ind = np.arange(N)    # the x locations for the groups
width = 0.35       # the width of the bars: can also be len(x) sequence

p1 = plt.bar(ind, A,width, color='r')

plt.ylabel('Fehler')
plt.title('Fehler verschiederner Verfahren')

plt.xticks(ind+width/2., ('Hallo', '2', '3', '4', '5', '6', '7', '8', '9', '10'))#dynamic - fed

# plt.yticks(np.arange(0,300,10))
# plt.legend( (p1[0], p2[0], p3[0]), ('A','B','C') )
# plt.grid(True)


plt.show()
