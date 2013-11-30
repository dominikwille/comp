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
def dat(x, delim = "\t", offset = 3):
    fn = os.path.join(os.path.dirname(__file__), x)

    data = np.genfromtxt(fn, delimiter = delim)
    
    x=np.empty([len(data) - offset])
    y=np.empty([len(data) - offset])
    j = 0
    for i in range(0,len(data) - offset):
        x[j]=data[j,0]
        y[j]=data[j,1]
        j += 1
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
    return np.exp(-a[0]*t)*(a[2]*np.sin(a[1]*t)+a[3]*np.cos(a[1]*t))

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

def ga(i, a, t):
    if(i == 0):
        return -t*np.exp(-a0*t)*a2*np.sin(a1*t)
    elif(i == 1):
        return np.exp(-a0*t)*a2*t*np.cos(a1*t)
    elif(i == 2):
        return np.exp(-a0*t)*np.sin(a1*t)
    else:
        return 0	
	

a1 = np.array([0.8, 6.4, 4.2,-0.3])
a2 = np.array([0.3, 5.4, 7.2,-1.3])
a3 = np.array([1.0, 7.0,-6.0, 3.0])
b1 = np.array([0.8, 6.4, 4.2])
b2 = np.array([0.3, 5.4, 7.2])
b3 = np.array([1.0, 7.0,-6.0])	
	




t = dat('data2', " ", 0)[0]
y = dat('data2', " ", 0)[1]
n = len(t)

m = 4




def Jac(funca, n, m, a, t):
    D = np.empty([n, m])
    for i in range(m):
        D[:,i] = np.vectorize(funca, excluded=[0, 1])(i, a, t)
    return np.matrix(D)


#iterate
def iterate(func, funca, n, m, a0, t, stop = 1e-6, damping = False):
    a = a0
    while(True):
        a_old = a
        g = np.matrix(y - np.vectorize(func, excluded=[0])(a, t))
        
        D = Jac(funca, n, m, a, t)
        delta = np.squeeze(np.asarray(np.linalg.solve(D.T*D, D.T*g.T)))

        while(damping):
            break
        a = a + delta
        print np.linalg.norm(a-a_old)
        if(np.linalg.norm(a-a_old) < stop):
            return a


a = iterate(f, fa, n, m, a1, t, 1e-8)
print a

