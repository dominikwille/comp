# -*- coding: utf-8 -*-


#Packete einbinden
import numpy as np

#LGS einlesen
F = np.matrix('1.,-2.,-1.;36000.,2.,0.;-2.,1400.,1.')
d = np.matrix('3.;72002.;1399.')
#Kontrollausgaben:
#print F
#print d


#3.1.1
#Vernichtungsfunktion
def eli(n,m):
    mod = F[n,m]/F[m,m]
    F[n] = F[n]-mod*F[m]
    d[n] = d[n]-mod*d[m]


#Stufenmatrix, liefert in erster Komponente die Stufenform und in der zweiten den
#gewandelten Ergebnissvektor
def stufenmatrix(F,d):
    for j in range (0,F.shape[0]-1):
        for i in range (j+1,F.shape[0]):
            eli(i,j)
    return F,d
        
#Kontrollausgaben
#print F
#print d

#Kontrollausgaben
#c=d[2]/F[2,2]
#print c
#b=(d[1]-F[1,2]*c)/F[1,1]
#print b
#c=(d[0]-F[0,1]*b-F[0,2]*c)/F[0,0]
#print c

#Berechnung des Lösungsvektors
def loesungsvektor(F,d):
    l = np.matrix ('0.;0.;0.')
    mod = 0
    for i in range(F.shape[0]-1, -1, -1):
        for j in range(i,F.shape[0]-1):
            mod = mod+F[i,j+1]*l[j+1]
        l[i] = (d[i]-mod)/F[i,i]
        mod = 0
    return l[::-1]

print loesungsvektor(stufenmatrix(F,d)[0],stufenmatrix(F,d)[1])

    
#3.1.2  
import math 
 
def mround (x):
    if ( x==0 ):
        return x
    return round (x, int (4 - math.ceil ( math.log10 ( abs(x )))))
    
vecmround = np.vectorize(mround)
    
    
F = vecmround(np.matrix('1.,-2.,-1.;36000.,2.,0.;-2.,1400.,1.'))
d = vecmround(np.matrix('3.;72002.;1399.'))

def eligleit(n,m):
    mod = mround(F[n,m]/F[m,m])
    F[n] = vecmround(F[n]-vecmround(mod*F[m]))
    d[n] = mround(d[n]-mround(mod*d[m]))
    
def stufenmatrixgleit(F,d):
    for j in range (0,F.shape[0]-1):
        for i in range (j+1,F.shape[0]):
            eligleit(i,j)
    return F,d
    
def loesungsvektorgleit(F,d):
    l = np.matrix ('0.;0.;0.')
    mod = 0
    for i in range(F.shape[0]-1, -1, -1):
        for j in range(i,F.shape[0]-1):
            mod = mround(mod+mround(F[i,j+1]*l[j+1]))
        l[i] = mround((mround(d[i]-mod))/F[i,i])
        mod = 0
    return l[::-1]

print loesungsvektorgleit(stufenmatrixgleit(F,d)[0],stufenmatrixgleit(F,d)[1])


#3.1.3
F = np.matrix('1.,-2.,-1.;36000.,2.,0.;-2.,1400.,1.')
d = np.matrix('3.;72002.;1399.')

def tauschen(i,j):
    val=abs(F[i,j])
    c=0
    for h in range(i+1,3):
        if abs(val) < abs(F[h,j]):
            val = F[h,j]
            index = h
            c = 1
    if c == 1:
        tmp = F[index].copy() 
        tmpd = d[index].copy() 
        F[index] = F[i].copy()
        d[index] = d[i].copy()
        F[i] = tmp.copy() 
        d[i] = tmpd.copy() 

def stufenmatrixgleitpiv(F,d):
    for j in range (0,F.shape[0]-1):
        for i in range (j+1,F.shape[0]):
            tauschen (j,j)
            eligleit(i,j)
    return F,d

print loesungsvektorgleit(stufenmatrixgleitpiv(F,d)[0],stufenmatrixgleitpiv(F,d)[1])