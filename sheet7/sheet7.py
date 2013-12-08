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

# Frage an den Tutor: is es möglich funktionen als parameter zu übergeben, und dabei einige
# parameter schon zu setzten. Ich habe jetzt fast 1h damit verschwendet zu googlen und bin
# noch zu keinem Schluss gekommen. Daher jetzt hier etwas eklig:
def error_h(func1, func2, values, func, h):
        err = 0.0
        for i in values:
                err += np.abs(func1(i) - func2(func, i, h))
        return err

def error_hk(func1, func2, values, func, h, k):
        err = 0.0
        for i in values:
                err += abs(func1(i) - func2(func, D, h, 0, k, i))
        return err

#1. verfahren
l = []
for i in range(0,6):
        l.append((error_h(Dana, D, x, f, 2.0**(-i)), 'Differenzenquotient h = $2^{-' + str(i) + '}$'))

#2. verfahren
for i in range(0,6):
        l.append((error_h(Dana, D2, x, f, 2.0**(-i)), 'Taylor h = $2^{-' + str(i) + '}$'))

#3. Verfahren
for i in range(0,6):
        l.append((error_hk(Dana, h_extrapolation, x, f, 1.0, i), 'h-Extrapolation h = 1; n = ' + str(i)))

l =  sorted(l, key=lambda tupel: tupel[0])

A = []
legend = []
for i in l:
        A.append(i[0])
        legend.append(i[1])

A = np.array(A)
N = len(A)

ind = np.arange(N)   
width = 0.35 

p1 = plt.bar(ind, A,width, color='r')

plt.ylabel('Fehler')
plt.title('Fehler verschiederner Differentiationsverfahren')

plt.xticks(ind+width/2., legend, rotation=90)
plt.tight_layout()
plt.show()








#7.2.2
#Definition der Verfahren:
#Rechteckverfahren:
def IRI(func,a,b):
	return func((a+b)/2)*(b-a)
#Trapezregel:
def ITI(func,a,b):
	return (b-a)*(func(a)+func(b))/2
#Simpsonregel:
def ISI(func,a,b):
	return (b-a)/6*(func(a)+4*func((a+b)/2)+func(b))
	
#def int(m, n, x):
	
