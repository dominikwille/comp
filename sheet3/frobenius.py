# -*- coding: utf-8 -*-

import numpy as np
import math

#3.3.1
#Eingabe der Daten
A=np.matrix('1.,-2.,-1.;36000.,2.,0.;-2.,1400.,1.')
A2=np.matrix([[1.,1/2.,1/3.],[1/2.,1/3.,1/4.],[1/3.,1/4.,1/5.]])
b=np.matrix('3.;72002.;1399.')
b2=np.matrix('0.;0.;0.')
b2[0]=A2[0,0]+A2[0,1]+A2[0,2]
b2[1]=A2[1,0]+A2[1,1]+A2[1,2]
b2[2]=A2[2,0]+A2[2,1]+A2[2,2]

#Frobenius-Funktion
def frobenius(A):
    sum=0.
    for i in range (0,len(A)):
        for j in range (0,len(A)):
            sum=sum+abs(A[i,j])**2
    return math.sqrt(sum)

#Euklidische Norm Funktion
def euknorm(b):
    sum=0
    for i in range(0,len(b)):
        sum=sum+b[i]**2
    return math.sqrt(sum)

#Ausgabe   
print 'Die berechneten Normen sind:'      
print frobenius(A)
print frobenius(A2)
print euknorm(b)
print euknorm(b2)
            
                                    

#3.3.2

#Dateneingabe
bf = np.array([[1.1],[0.9],[1.05]])
b2f = np.array([[1.1],[0.9],[1.05]])
lam = np.array([[1.],[0.1],[5]])
for i in range(0,3):
    bf[i]=bf[i]*b[i]
for i in range(0,3):
    b2f[i]=b2f[i]*b2[i]
    

#Koordinationszahl
def coord(M):
    c=frobenius(M)*frobenius(np.linalg.inv(M))
    return c
    
#Abweichungen
print 'Die Abweichungen betragen:'
for i in range (0,3):
    print euknorm(b-lam[i]*bf)
print 'für b und'
for i in range (0,3):
    print euknorm(b2-lam[i]*b2f)
print 'für bstrich'

#relativer Fehler
print 'der relative Fehler ergibt sich zu:' 
for i in range (0,3):
    print coord(A)*euknorm(b-lam[i]*bf)/euknorm(b)
print 'für b und'
for i in range (0,3):
    print coord(A)*euknorm(b2-lam[i]*b2f)/euknorm(b2)
print 'für bstrich'

#Lösungen der LGS:
print 'Die Lösungsvektoren der LGS ergeben sich zu:'
print np.linalg.solve(A,b)
print 'und'
print np.linalg.solve(A2,b2)

#absolute Fehler
print 'damit ergeben sich die Absoluten Fehler zu:'
for i in range (0,3):
    print coord(A)*euknorm(b-lam[i]*bf)/euknorm(b)*euknorm(np.linalg.solve(A,b))
print 'für b und'
for i in range (0,3):
    print coord(A)*euknorm(b2-lam[i]*b2f)/euknorm(b2)*euknorm(np.linalg.solve(A2,b2))
print 'für bstrich'

