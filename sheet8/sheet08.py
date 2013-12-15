# -*- coding: utf-8-sig -*-
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.setrecursionlimit(10000000)


#8.1 a)
#Die analytische Lösung der Gleichung lautet e^(-t).

def f(t,y):
	return 3 * y - 4 * np.exp(-t)

	
def euler(func, h, ts, tm, n, y, i = [], j = []):
    i.append(ts + n * h)
    j.append(y)  
    if ts + n * h <= tm:
        y = y + h * func(ts + n * h, y)
        euler(func, h, ts, tm, n+1, y, i, j)
    else:
        del i[-1]
        del j[-1]
    return i,j	            
        
   
def mittelp(func, h, ts, tm, n, y, i, j):		
    i.append(ts + n * h)
    j.append(y)
    if ts + n * h <= tm:
        y = y + h * func(ts + n * h + h/2., y + h/2 * func(ts + n * h, y))
        mittelp(func, h, ts, tm, n+1, y, i, j)
    else:
        del i[-1]
        del j[-1]
    return i,j	            
        

def runge(func, h, ts, tm, n, y, i, j):
    i.append(ts + n * h)
    j.append(y)
	
    A = func(ts + n * h, y)
    B = func(ts + n * h + h/2., y + h/2. * A)
    C = func(ts + n * h + h/2., y + h/2. * B)
    D = func(ts + n * h + h, y + h * C)
	
    if ts + n * h <= tm:
	y = y + h/6. * (A + 2*B + 2*C + D)
	runge(func, h, ts, tm, n+1, y, i, j) 
    else:
        del i[-1]
        del j[-1]
    return i,j	



#plot: wurde als figure_1.png exportiert.
#z = np.exp(-np.asarray(euler(f, 0.001, 0., 2., 0.,1., [], [])[0]))


#plt.plot(euler(f, 0.1, 0., 2., 0.,1., [], [])[0], euler(f, 0.1, 0., 2., 0.,1., [], [])[1])
#plt.plot(mittelp(f, 0.1, 0., 2., 0.,1., [], [])[0], mittelp(f, 0.1, 0., 2., 0.,1., [], [])[1])
#plt.plot(runge(f, 0.1, 0., 2., 0.,1., [], [])[0], runge(f, 0.1, 0., 2., 0.,1., [], [])[1])


#plt.plot(euler(f, 0.01, 0., 2., 0.,1., [], [])[0], euler(f, 0.01, 0., 2., 0.,1., [], [])[1])
#plt.plot(mittelp(f, 0.01, 0., 2., 0.,1., [], [])[0], mittelp(f, 0.01, 0., 2., 0.,1., [], [])[1])
#plt.plot(runge(f, 0.01, 0., 2., 0.,1., [], [])[0], runge(f, 0.01, 0., 2., 0.,1., [], [])[1])


#plt.plot(euler(f, 0.001, 0., 2., 0.,1., [], [])[0], euler(f, 0.001, 0., 2., 0.,1., [], [])[1],'--')
#plt.plot(mittelp(f, 0.001, 0., 2., 0.,1., [], [])[0], mittelp(f, 0.001, 0., 2., 0.,1., [], [])[1],'--')
#plt.plot(runge(f, 0.001, 0., 2., 0.,1., [], [])[0], runge(f, 0.001, 0., 2., 0.,1., [], [])[1],'--')

#plt.plot(runge(f, 0.001, 0., 2., 0.,1., [], [])[0], z, ':')


#plt.legend( ('euler 0.1','mittelp 0.1','runge 0.1', 'euler 0.01','mittelp 0.01','runge 0.01', 'euler 0.001','mittelp 0.001','runge 0.001','analytisch'), loc=3)

#plt.xlabel('t')
#plt.ylabel('y')

#plt.show()



#8.1 b)



print 'Die Fehler fuer das Euler-Verfahen lauten:'
print abs(euler(f, 0.1, 0., 2., 0.,1., [], [])[1][10]-np.exp(-1))
print abs(euler(f, 0.01, 0., 2., 0.,1., [], [])[1][100]-np.exp(-1))
print abs(euler(f, 0.001, 0., 2., 0.,1., [], [])[1][1000]-np.exp(-1))

print 'Die Fehler fuer das Mittelpunktverfahen lauten:'
print abs(mittelp(f, 0.1, 0., 2., 0.,1., [], [])[1][10]-np.exp(-1))
print abs(mittelp(f, 0.01, 0., 2., 0.,1., [], [])[1][100]-np.exp(-1))
print abs(mittelp(f, 0.001, 0., 2., 0.,1., [], [])[1][1000]-np.exp(-1))

print 'Die Fehler fuer das Runge-Kutta Verfahren lauten:'
print abs(runge(f, 0.1, 0., 2., 0.,1., [], [])[1][10]-np.exp(-1))
print abs(runge(f, 0.01, 0., 2., 0.,1., [], [])[1][100]-np.exp(-1))
print abs(runge(f, 0.001, 0., 2., 0.,1., [], [])[1][1000]-np.exp(-1))


#plot: wurde als figure_2.png exportiert.
#def ploteuler(x):
#	return euler(f, x, 0., 2., 0.,1., [], [])[1][10]

#def plotmittelp(x):
#	return mittelp(f, x, 0., 2., 0.,1., [], [])[1][10]

#def plotrunge(x):
#	return runge(f, x, 0., 2., 0.,1., [], [])[1][10]

#pe = np.vectorize(ploteuler)
#pm = np.vectorize(plotmittelp)
#pr = np.vectorize(plotrunge)

#h = np.arange(0.001,0.1,0.0001)

#plt.loglog(h, pe(h), h, pm(h), h, pr(h))

#plt.legend( ('Euler', 'Mittelp', 'Runge'), loc=3)
#plt.show()
 

