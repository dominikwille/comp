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
import os


#6.1
#Daten einlesen:
def dat(x, del = "\t", offset = 0):
    fn = os.path.join(os.path.dirname(__file__), x)

    data = np.genfromtxt(fn, delimiter = "\t")

    x=np.empty([len(data)])
    y=np.empty([len(data)])
    for i in range(0,len(data)):
        x[i]=data[i,0]
        y[i]=data[i,1]
    return x,y

#Linere Ausgleichsrechnung	
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

#Belegung der Daten
x = dat('data')[0]
y = dat('data')[1]
n = len(x)
 
 
# #Berechnung der Koefizienten
# a_max = 3
# A = np.empty([n,a_max])

# for i in range(a_max):
# 	A[:,i] = np.vectorize(f1)(x, i)
	
	
# A = np.matrix(A)
# y = np.matrix(y)
# a = np.linalg.solve(A.T*A,A.T*y.T)
# print a

# b_max = 3
# B = np.empty([n,b_max])
# for i in range(a_max):
#     B[:,i] = np.vectorize(f2)(x, i)

# B = np.matrix(B)
# b = np.linalg.solve(B.T*B,B.T*y.T)
# print b


# #Definition der Fits 
# def F1(x):
# 	y = 0
# 	for i in range(len(a)):
# 		y += a[i]*f1(x,i)
# 	return y

# def F2(x):
# 	y = 0
# 	for i in range(len(b)):
# 		y += b[i]*f2(x,i)
# 	return y

	
# x = dat('data')[0]
# y = dat('data')[1]	
# sum = 0


# #Fehlerrechnung
# for i in range(n):
# 	sum +=(y[i]-F1(x[i]))**2

# print 'Abweichung für erste Ansatzfunktion:' 
# print sum

# for i in range(n):
# 	sum +=(y[i]-F2(x[i]))**2
# print 'Abweichung für zweite Ansatzfunktion:'
# print sum
       

# max = 0
# for i in range(n):
# 	check = (y[i]-F1(x[i]))**2
# 	if check > max:
# 		max = check
# 		index = i
# print 'Maximaler Fehler für erste Ansatzfunktion:'
# print max
# print 'Dieser tritt im folgenden Jahr auf::'
# print x[index]

# max = 0
# for i in range(n):
# 	check = (y[i]-F2(x[i]))**2
# 	if check > max:
# 		max = check
# 		index = i
# print 'Maximaler Fehler für zweite Ansatzfunktion:'
# print max
# print 'Dieser tritt im folgenden Jahr auf::'
# print x[index]
	   

# #Plot
# x = np.arange(1970.,2030.,1.)
# plt.plot(dat('data')[0], dat('data')[1], 'bs', x, np.vectorize(F1)(x), x, np.vectorize(F2)(x))
# plt.legend(('Daten', '$(i)$', '$(ii)$'), loc=1)
# plt.xlabel('Jahr')
# plt.ylabel('Doenerpreis in Euro')
# #plt.show()



#6.2
#Funktionen und partielle Ableitungen
def f(a, t):
	return np.exp(-a[0]*t)*(a[2]*np.sin(a[1]*t)+a3*np.cos(a[1]*t))

def fa(i, a, t):
    if(i == 0):
        return -t*np.exp(-a[0]*t)*(a[2]*np.sin(a[1]*t)+a[3]*np.cos(a[1]*t))
    elif(i == 1):
        return t*np.exp(-a[0]*t)*(a[2]*np.cos(a[1]*t)-a[3]*np.sin(a[1]*t))
    elif(i == 2):
        return np.exp(-a[0]*t)*np.sin(a[1]*t)
    elif(i == 3):
        return np.exp(-a[0]*t)*np.cos(a[1]*t)
    else:
        return 0

def g(a0,a1,a2,t):
	return np.exp(-a0*t)*a2*np.sin(a1*t)

def ga0(a0,a1,a2,t):
	return -t*np.exp(-a0*t)*a2*np.sin(a1*t)
	
def ga1(a0,a1,a2,t):
	return np.exp(-a0*t)*a2*t*np.cos(a1*t)
	
def ga2(a0,a1,a2,t):
	return np.exp(-a0*t)*np.sin(a1*t)
	
a1 = np.array([0.8, 6.4, 4.2,-0.3])
a2 = np.array([0.3, 5.4, 7.2,-1.3])
a3 = np.array([1.0, 7.0,-6.0, 3.0])
b1 = np.array([0.8, 6.4, 4.2])
b2 = np.array([0.3, 5.4, 7.2])
b3 = np.array([1.0, 7.0,-6.0])	
	




t = dat('data2', " ", 1)[0]
y = dat('data2', " ", 1)[1]
n = len(x)

k = 4


#set a0
a = a1

def Jac(n, k, a, t):
    D = np.empty([n, k])
    for i in range(k):
        D[:,i] = np.vectorize(fa)(i, a, t)
    return np.matrix(D)


#iterate
while(True):
    a_old = a
    g = np.vectorize(f)(t) - y
    D = Jac(n, k, a, t)
    a = np.linalg.solve(D.T*D, D.T*g)
    
    if(np.linalg.norm(a-a_old) < 1e-6):
        break;

