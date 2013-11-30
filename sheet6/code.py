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
import os


#6.1
#Daten einlesen:
def dat(x):
    fn = os.path.join(os.path.dirname(__file__), x)

    data = np.genfromtxt(fn, delimiter = "\t")

    x=np.empty([len(data)])
    y=np.empty([len(data)])
    for i in range(0,len(data)):
        x[i]=data[i,0]
        y[i]=data[i,1]
    return x,y

#Definitionen der Teilfunktionen
def f1(x, f):
    if(f = 1):
        return ((x-1970.)/100.)**2.
    elif(f = 2):
        return ((x-1970.)/100.)
    elif(f = 3):
        return 1
    else:
        except('Falscher Funktionsaufruf')

#Todo: refactor this....
#
# def f21(x):
#     return ((x-1970.)/100.)**3.

# f22 = f12

# def f23(x):
#     return np.cos((x-1970)/100)

x = dat('data')[0]
y = dat('data')[1]
n = len(x)
 
A = np.empty([n,3])
print np.vectorize(f12)(x)
   




       
