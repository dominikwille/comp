#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @autor Dominik Wille
# @tutor Alexander Schlaich
# @exercise 1.1
#
# This is my very small script to calculate the length of the mantissa in
# different cases. The script will start at some value a and devide it
# by 2 until it doesn't chnage any longer. The values of a are set up as list.

# Import math-Modulte so do some log calculations.
import math

#List of a values that should be calculated.
list = [1.0, 1.1*10**(-5), 10**(-30), 5*10**(12)]

#First value of e
e = 1.0

#The results-list will be filled with tupels (a, e).
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


print 'Exeecise 1.1.1:\n'

#print all results
for (a, e) in results:
    l = len(str(a))
    fill = (maxlen - l) * ' '
    man = int(-math.log(e, 2) - 1)
    # man = man * 2 + 2
    # man = math.log(man, 2)
    man1 = math.frexp(e)
    print '    a=' + str(a) + fill + ' e=' + str(e) + ' => ' + str(man1) + str(man)

print '\nExercise 1.1.2:\n'
print '    e is the machine epsilon.'

print '\nExercise 1.1.3:\n'
print '    The value can be calculated. it is the smallest mantissa (000000...) and the smallest exponent'
print '    (00000...). For 32-Bit this means 0.5 * 10**(-7) = ' + str(0.5 * 10**(-7))
