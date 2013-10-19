#!/usr/local/bin/python
# -*- coding: utf-8 -*-

def area(a, b):
    if ( a >= 0 and b >= 0 ):
        return a * b
    else:
        return -1

def isfloat(x):
    try:
        float(x)
        return True
    except:
        return False

def isint(x):
    try:
        y = int(x)
    except:
        return False
    if(x == str(y)):
        return True
    return False

def print_area (a, b):
    c = area(a, b)
    if(c >= 0):
        return str(c)
    return 'Fehler bei der Berechnung - überprüfen sie ihre Eingaben'

def input_number(str):
    x = 'foo'
    first = True
    while(not isfloat(x) or x < 0):
        if(first):
            first = False
        else:
            print 'Das ist keine Zahl!'
        x = raw_input(str)
    return float(x)

def input_unsigned_integer(str):
    x = 'foo'
    first = True
    while(not x.isdigit()):
        if(first):
            first = False
        else:
            print 'Das ist keine natürliche Zahl!'
        x = raw_input(str)
    return x

def input_integer(str):
    x = 'foo'
    first = True
    while(not isint(x)):
        if(first):
            first = False
        else:
            print 'Das ist keine ganze zahl!'
        x = raw_input(str)
    return x

def faculty(n):
    if(n <= 1):
        return 1;
    else:
        return faculty(n - 1) * n
        
def print_faculty(n):
    if(n.isdigit() and n >= 0):
        return str(faculty(int(n)))
    else:
        return 'Fehler bei der Berechnung - überprüfen sie ihre Eingaben'


n = input_integer('geben sie eine Zahl ein: ')

print "Die fakultät von %s ist: %s" % (n, print_faculty(n))

if(1):
    print '1 ist wahr'
if(1.1):
    print '1.1 ist wahr'
if(0.0):
    print '0.0 ist nicht wahr'
