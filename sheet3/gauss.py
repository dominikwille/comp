# -*- coding: utf-8 -*-
#Packete einbinden
import numpy as np

#LGS einlesen
F = np.matrix('1.,-2.,-1.;36000.,2.,0.;-2.,1400.,1.')
d = np.matrix('3.;72002.;1399.')

#Vernichtungsfunktion
def eli(n,m):
    mod = F[n,m]/F[m,m]
    F[n] = F[n]-mod*F[m]
    d[n] = d[n]-mod*d[m]


#Vernichtungsfunktion schleift über LGS
for j in range (0,F.shape[0]-1):
    for i in range (j+1,F.shape[0]):
        eli(i,j)
        
#Ausgabe Stufenform
print F
print d