#!/usr/local/bin/python
# -*- coding: utf-8 -*-

import sys

def area (a, b):
    if ( a >= 0 and b >= 0 ):
        return a * b
    else:
        return -1;

def isfloat (x):
    try:
        float(x);
        return True
    except:
        return False

def print_area (a, b):
    c = area(a, b)
    if(c >= 0):
        return str(c)
    return 'Fehler bei der Berechnung - überprüfen sie ihre Eingaben'

def input_n (str):
    x = 'foo'
    first = True;
    while(not isfloat(x) or x < 0):
        if(first):
            first = False
        else:
            print 'Das ist keine Zahl!'
        x = raw_input(str);
    return float(x)

a = b = -1

inp = sys.argv

if(len(inp) >= 2 and isfloat(inp[1])):
    a = float(inp[1])
if(len(inp) >= 3 and isfloat(inp[2])):
    b = float(inp[2])

if(a < 0):
    a = input_n('Länge a des Rechtecks: ')
if(b < 0):
    b = input_n('Länge b des Rechtecks: ')

print "das Rechteck hat einen Flächeninhalt von: %s" % (print_area(a, b))
