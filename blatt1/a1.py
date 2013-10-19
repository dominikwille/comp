#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @autor Dominik Wille
# @tutor Alexander Schlaich
# @exercise 1.1

import math

e = 1.0
list = [1.0, 1.1*10**(-5), 10**(-30), 5*10**(12)]
results = []

for a in list: 
    while(not a == a + e):
        e = e * 0.5
    results.append((a, e))

#-------OUTPUT--------

#prepare for output - get maximal strlen of a
maxlen = 0
for (a, e) in results:
    if(len(str(a))) > maxlen:
        maxlen = len(str(a))


print 'Aufgabe 1.1.1:\n'

#print all results
for (a, e) in results:
    l = len(str(a))
    fill = (maxlen - l) * ' '
    man = int(-math.log(e, 2) - 1)
    print '    a=' + str(a) + fill + ' e=' + str(e) + ' das entspricht einer ' + str(man) + '-stelligen Mantisse'


# a=1.0           e=1.11022302463e-16
# a=1.1e-05       e=8.47032947254e-22
# a=1e-30         e=8.75811540203e-47
# a=5000000000000 e=8.75811540203e-47



print '\nAufgabe 1.1.2:\n'
print '    Der Wert von e repräsentiert den kleinst möglichen relativen Abstand zweier Zahlen'

print '\nAufgabe 1.1.3:\n'
print '    Ja kann man, er entspricht dem kleinsten in der Mantisse darstellbaren Wert.'
print '    Für eine 32-Bit single precision Mantisse sind 23 Bits vorgesehn, d.h. die kleinste darstellabre Zahl ist: \n'
print '    2**-(23+1) = ' + str(2**-(23+1))
print '\n    Die eine 24ste Stelle rührt daher, dass die esrste Stelle der Mantisse sonst imer 1 wäre.'
