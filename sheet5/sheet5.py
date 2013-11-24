# -*- coding: utf-8 -*-
#Packete
import numpy as np
import os
import matplotlib.pyplot as plt
#Daten einlesen, global:x,y,n:
fn = os.path.join(os.path.dirname(__file__), 'daten.txt')
data = np.genfromtxt(fn, delimiter = ' ')
x=np.empty([len(data)])
y=np.empty([len(data)])
n=len(x)
for i in range(0,n):
    x[i]=data[i,0]
    y[i]=data[i,1]
    
    
#5.1.1    
#Erstellung der x-Matrix:
M = np.empty([n, n])
for i in range(0,n):
    M[i] = x[i]
    for j in range(0,n):
        M[i,j] = M[i,j]**j

#Das interpolierte Polynom:
def poly(z):
    a = np.linalg.solve(M,y)
    sum = 0
    for i in range(0,n):
        sum+=a[i]*z**i
    return sum
    
    
#5.1.2
#Berechnung der Faktoren:
def l(z,i):
    prod = 1
    for j in range(0,n):
        if i!=j:
            prod=prod*(z-x[j])/(x[i]-x[j])
    return prod

#Das interpolierte Polynom:
def polyl(z):
    sum=0
    for i in range(0,n):
        sum = sum+y[i]*l(z,i)
    return sum
   
     
#5.1.3
#Dividierte Differenzen, global:dd:
def f(x,y):
    a = np.copy(y)
    for k in range(1,n):
        for i in range(n-1,k-1,-1):
            a[i]=(a[i]-a[i-1])/(x[i]-x[i-k])
    return a
dd = np.copy(f(x,y))

#Definition des Polynoms:
def nj(z,i):
    prod=1
    for j in range(0,i):
        prod*=(z-x[j])
    return prod

def polyd(z):
    sum=0
    for i in range(0,n):
        sum+=dd[i]*nj(z,i)
    return sum

#Plot
x1=np.arange(1,13.1,0.01)
vpoly=np.vectorize(poly)
vpolyl=np.vectorize(polyl)
vpolyd=np.vectorize(polyd)
y1=vpoly(x1)
y2=vpolyl(x1)
y3=vpolyd(x1)
plt.plot(x1,y1,x1,y2,x1,y3,x,y,'bx')
#plt.show()
#Der Plot wurde als figure_1.png exportiert, wie dort zu sehen ist unterscheiden
#die verschiedenen Verfahren in dem geplotteten Bereich nicht.

#5.1.4
n=4
a=np.array([-1.,0.,1.,2.])
b=np.array([5.,-2.,9.,-4.])


#Berechnung der dividierten Differenz für (x_i,x_i+1,x_i+2)
def d(x,y,i):
    z = (b[i+1]-b[i])/(a[i+1]-a[i]) 
    i+=1
    z2 = (b[i+1]-b[i])/(a[i+1]-a[i]) 
    z3 = (z2-z)/(a[i+1]-a[i-1])
    return z3

#Berechnung mu(i):
def mu(i):
    m=(a[i]-a[i-1])/(a[i+1]-a[i-1])
    return m



