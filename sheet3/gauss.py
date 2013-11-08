# -*- coding: utf-8 -*-


#Packete einbinden
import numpy as np

#LGS einlesen
F = np.matrix('1.,-2.,-1.;36000.,2.,0.;-2.,1400.,1.')
d = np.matrix('3.;72002.;1399.')
#Kontrollausgaben:
#print F
#print d

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