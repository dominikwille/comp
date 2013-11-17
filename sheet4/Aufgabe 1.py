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

