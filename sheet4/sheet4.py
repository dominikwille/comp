#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 4
#
# Bitte die plots zum testen einkommentieren.

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

np.seterr(all='ignore')



#4.1.1
#Zur Lösung mit dem Gauß-Seidel Verfahren werden zunächst Funktionen,
#zum Umformen der eingegebenen Daten implementiert und Schließlich
#gaussseidel(L,D,R,b,k), b beschreibt hierbei das Konvergenzkriterium
#Für k != 0 gibt k die zu vernwendende Anzahl der Iterationsschritte an.
#Es werden für die gegebene Matrix 12 Iterationsschritte bnötigt.

#Einlesen der Daten
a = np.matrix([[3.5, 3., -0.5], [-1, 4, 4],  [1./3., 3., 4.5]])
b = np.matrix([[7.5],[-6.5],[1.]])

#Funktionen zum Umformen der Daten
def untermat(a):
    du = np.zeros(shape=(len(a),len(a)))    
    for i in range (0, len(a)):
        for j in range (0, len(a)):
            if i>j:
                du[i,j]=a[i,j]
    return du
 
def obermat(a):
    do = np.zeros(shape=(len(a),len(a)))                         
    for i in range (0, len(a)):
        for j in range (0, len(a)):
            if i<j:
                do[i,j]=a[i,j]
    return do
    
def diagmat(a):
    d = np.zeros(shape=(len(a),len(a)))                         
    for i in range (0, len(a)):
        for j in range (0, len(a)):
            if i==j:
                d[i,j]=a[i,j]
    return d    

#Umformung der Daten
R = obermat(a)
L = untermat(a)
D = diagmat(a)

#Gauß-Seidel
def gaussseidel(L,D,R,b,k):
    X = np.zeros(shape=(len(a),1))
    
    #Hilfsfunktionen
    def rechteseite(R,X,b,i):
        return -R*np.matrix(X[:,i]).reshape((len(X), 1))+b

    def einsetzen(F,d):
        l = np.zeros(shape=(d.shape[0],1))
        mod = 0
        for i in range (0,d.shape[0]):
            for j in range(0,i+1):
                mod = mod+F[i,j]*l[j]
            l[i] = (d[i]-mod)/F[i,i]
            mod = 0 
        return l

    #Auswertung Konv-Kriterium
    if k != 0:
        i = 0
        konv=10
        while  konv>k:
            X = np.concatenate((X,einsetzen(D+L,rechteseite(R,X,b,i))),axis=1)
            konv = np.linalg.norm(X[:,i-1]-X[:,i])
            i+=1
            
    else:
    #Auswertung nach Iterationsschritten
        for i in range(0,500):
            X = np.concatenate((X,einsetzen(D+L,rechteseite(R,X,b,i))),axis=1)
    #Ausgabe
    print 'Eine Auswertung mit dem Gauss-Seidel-Verfahren führte zu folgenden Werten:'
    print 'Lösungsvektor:'
    X = X[:,i]
    print np.round(X,4)
    print 'Anzahl der Schritte:' 
    print i+1
print 'Aufgabe 4.1.1:'
gaussseidel(L,D,R,b,0.5E-4)


#4.1.2
#Die Übergabewerte von jacobi(L,D,R,b,k) sind die selben wie in der
#vorherigen Aufgabe, die Anzahl der Iterationsschritte hat sich leicht erhöht
#während das Ergebniss auf 4 Dezimalstellen das selbe bleibt.


def jacobi(L,D,R,b,k):
    X = np.zeros(shape=(len(a),1))
    
    #Hilfsfunktionen
    def rechteseite(R,X,b,i):
        return (-L-R)*np.matrix(X[:,i]).reshape((len(X), 1))+b

    def einsetzen(F,d):
        l = np.zeros(shape=(d.shape[0],1))
        for i in range (0,d.shape[0]):
            l [i] = d[i]/F[i,i]
        return l

    #Auswertung Konv-Kriterium
    if k != 0:
        i = 0
        konv=10
        while  konv>k:
            X = np.concatenate((X,einsetzen(D,rechteseite(R,X,b,i))),axis=1)
            konv = np.linalg.norm(X[:,i-1]-X[:,i])
            i+=1
    else:
    #Auswertung nach Iterationsschritten
        for i in range(0,500):
            X = np.concatenate((X,einsetzen(D,rechteseite(R,X,b,i))),axis=1)
    
    #Ausgabe
    print 'Eine Auswertung mit dem Jacobi-Verfahren führte zu folgenden Werten:'
    print 'Lösungsvektor:'
    X = X[:,i]
    print np.round(X,4)
    print 'Anzahl der Schritte:' 
    print i+1
print 'Aufgabe 4.1.2:'
jacobi(L,D,R,b,0.5E-4)

#4.1.3
#Das Verfahren konvergiert für jede Matrix für die gilt: -(D+L)**(-1)*U<1

#4.1.4
#Die Gleichung lässt sich nur mit dem Gauß-Seidel-Verfahren lösen.
#Einlesen der neuen Daten
a = np.matrix([[5., 3., -1.,2.], [-3., 7., 6., -2.],  [4., 4., 3., -3.],  [-5., 2., 2., 4.]])
b = np.matrix([[8.],[1.],[7.],[2.]])

#Umformung der Daten
R = obermat(a)
L = untermat(a)
D = diagmat(a)


#Lösung des Systems
print 'Aufgabe 4.1.4:'
jacobi(L,D,R,b,0)
gaussseidel(L,D,R,b,0)


#4.1.5
import matplotlib.pyplot as plt
#from scipy.signal import argrelextrema
#Zum Plotten wurde eine neue Funktion definiert und der entstandene Plot wurde als
#Figure 1 exportiert. Am Plot lässt sich erkennen, dass das Verfahren für alle
#Werte in (0,2) mit ausnahme eines w-Wertes bei ca 1,1. 


def gaussseidelrelax(w):
    a = np.matrix([[5., 3., -1.,2.], [-3., 7., 6., -2.],  [4., 4., 3., -3.],  [-5., 2., 2., 4.]])
    b = np.matrix([[8.],[1.],[7.],[2.]])
    R = obermat(a)
    L = untermat(a)
    D = diagmat(a)
    X = np.zeros(shape=(len(a),1))
    
    #Hilfsfunktionen
    def rechteseite(R,X,b,i):
        return ((1./w)*D-D-R)*np.matrix(X[:,i]).reshape((len(X), 1))+b

    def einsetzen(F,d):
        l = np.zeros(shape=(d.shape[0],1))
        mod = 0
        for i in range (0,d.shape[0]):
            for j in range(0,i+1):
                mod = mod+F[i,j]*l[j]
            l[i] = (d[i]-mod)/F[i,i]
            mod = 0 
        return l

    
    #Auswertung Konv-Kriterium
    i = 0
    konv=10
    while  konv>0.5E-4:
        X = np.concatenate((X,einsetzen((1./w)*D+L,rechteseite(R,X,b,i))),axis=1)
        konv = np.linalg.norm(X[:,i-1]-X[:,i])
        i+=1
    
    #Ausgabe
    return i+1

#Plot:
#xp=np.arange(0.01,2,0.01)
#vgaussseidelrelax=np.vectorize(gaussseidelrelax)
#yp=vgaussseidelrelax(xp)
#plt.plot(xp,yp)
#plt.show()
#print argrelextrema(yp, np.greater)
#for i in range(0,len(argrelextrema(yp, np.greater))):
#        print xp[argrelextrema(yp, np.greater)]

#4.2
print "\n\nAufgabe 2: "
def phi(x, y):
    return x**4 - x**2 + y**4 - 0.2 * y**3 - y**2 + 0.2 * x * y**3

def F(x, y):
    return np.matrix([-4.0*x**3 + 2.0*x - 0.2*y**3, -4.0*y**3 + 0.6*y**2 + 2.0*y - 0.6*x*y**2])

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

def key(l, x0, y0):
    for (x, y, key) in l:
        if(x == x0 and y == y0):
            return key
    return -1
    

def ex_plot(x, y, X, Y, extremum):
    points = []
    minima = []
    minimum = 1.0
    for x in X:
        for y in Y:
            z = extremum(x, y)
            x_max = z[0]
            y_max = z[1]
            H = np.matrix([[-f1x(x_max, y_max), -f1y(x_max, y_max)], [-f2x(x_max, y_max), -f2y(x_max, y_max)]])
            if(neg_def(H)):
                points.append((x, y , 5))
            elif(pos_def(H)):
                if(key(minima , round(x_max, 2), round(y_max, 2)) == -1):
                    minima.append((round(x_max, 2), round(y_max, 2), minimum))
                    points.append((x, y, minimum))
                    minimum += 1.0
                else:
                    points.append((x, y, key(minima, round(x_max, 2), round(y_max, 2))))
            else:
                points.append((x, y, 0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for(x, y, z) in points:
        ax.scatter(x, y, z)
    plt.show()

X = np.arange(-1.0, 1.01, 0.1)
Y = np.arange(-1.0, 1.01, 0.1)

#4.2.1
# X, Y = np.meshgrid(X, Y)
# Axes3D(plt.figure()).plot_wireframe(X, Y, np.vectorize(phi)(X, Y))
# plt.show()

#4.2.2
def extremum(x, y, n = 100):
    for i in range(0, n):
        a = np.array([[f1x(x, y), f1y(x, y)], [f2x(x, y), f2y(x, y)]])
        b = np.array([-f1(x, y), -f2(x, y)])

        z = np.linalg.solve(a, b)
        x += z[0]
        y += z[1]
    return [x, y]

#setze Startwerte
x = 0.05
y = 0.05
print "A4.2.2:"
print extremum(x, y)

#4.2.3
ex_plot(x, y, X, Y, extremum)

#4.2.4
def iteration(x, y, d=0.001, e=0.01):
    while(np.linalg.norm(F(x, y)) > d):
        z = F(x, y)
        x += e*z[0,0]
        y += e*z[0,1]

    return [x, y]
print "A4.2.4:"
print iteration(x, y)

#4.2.5
# ex_plot(x, y, X, Y, iteration)

#4.2.6
# Man könnte einfach epsilon * F(x,y) subtrahieren anststatt es zu addieren. Der
# momentan verwendete Algorithmus findet nur Minima! Ist der statwert ein extremum,
# so bleibt der wert kosntant, deshalb sind auch vereizelt maxima und sattelpunkte
# zu sehen.

