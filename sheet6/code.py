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
    if(f == 2):
        return ((x-1970.)/100.)**2
    elif(f == 1):
        return ((x-1970.)/100.)
    elif(f == 0):
        return 1
    else:
        return 0


def f2(x, f):		
    if(f == 2):
        return ((x-1970.)/100.)**2.
    elif(f == 1):
        return ((x-1970.)/100.)
    elif(f == 0):
        return np.cos((x-1970)/100)
    else:
        return 0




	


	
x = dat('data')[0]
y = dat('data')[1]
n = len(x)
 

a_max = 3
A = np.empty([n,a_max])


for i in range(a_max):
	A[:,i] = np.vectorize(f1)(x, i)
	
	
A = np.matrix(A)
y = np.matrix(y)
a = np.linalg.solve(A.T*A,A.T*y.T)
print a


b_max = 3
B = np.empty([n,b_max])
for i in range(a_max):
    B[:,i] = np.vectorize(f2)(x, i)

B = np.matrix(B)
b = np.linalg.solve(B.T*B,B.T*y.T)
print b

   
def F1(x):
	y = 0
	for i in range(len(a)):
		y += a[i]*f1(x,i)
	return y

def F2(x):
	y = 0
	for i in range(len(b)):
		y += b[i]*f1(x,i)
	return y
	

	
	
#sum = 0
#for i in range(n):
#	sum +=
       
