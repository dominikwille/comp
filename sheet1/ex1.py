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
    print '    a=' + str(a) + fill + ' e=' + str(e) + ' => ' + str(man) + '-Bit mantissa'

# My machine gives me the following output:
#
# a=1.0           e=1.11022302463e-16 => 52-Bit mantissa
# a=1.1e-05       e=8.47032947254e-22 => 69-Bit mantissa
# a=1e-30         e=8.75811540203e-47 => 152-Bit mantissa
# a=5000000000000 e=8.75811540203e-47 => 152-Bit mantissa
#
# that's kind of strange, never heard of a 69- or 152-Bit mantissa...


print '\nExercise 1.1.2:\n'
print '    The value of e is the smallest relative difference.'

print '\nExercise 1.1.3:\n'
print '    Yes that\'s possible, it is the smallest value the mantisse can represent.'
print '    For a 32-Bit single precision float the mantissa is 23 Bits long, that means the smalest value is: \n'
print '    2**-(23+1) = ' + str(2**-(23+1))
print '\n    The extra bit is caused by the fact that every mantissa would start with 1 otherwise.'
