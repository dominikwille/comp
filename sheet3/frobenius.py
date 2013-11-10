# -*- coding: utf-8 -*-

import numpy as np
import math

#3.3.1
#Eingabe der Daten
A=np.matrix('1.,-2.,-1.;36000.,2.,0.;-2.,1400.,1.')
b=np.matrix('3.;72002.;1399.')

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
    

#3.3.2