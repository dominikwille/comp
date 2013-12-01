#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 6
#
#Packete einlesen:
import numpy as np
import matplotlib.pyplot as plt
import os


#Daten einlesen Hilfsfunktion:
def dat(x, delim = "\t", offset = 0):
    fn = os.path.join(os.path.dirname(__file__), x)

    data = np.genfromtxt(fn, delimiter = delim)
    
    x=np.empty([len(data) - offset])
    y=np.empty([len(data) - offset])
    j = 0
    for i in range(0,len(data) - offset):
        x[j]=data[j,0]
        y[j]=data[j,1]
        j += 1
    return x,y



#6.1
print 'Aufgabe 6.1'
#Linere Ausgleichsrechnung	
#Definitionen der Teilfunktionen
def f1(x, f):
    if(f == 2):
        return ((x-1970.)/100.)**2
    elif(f == 1):
        return ((x-1970.)/100.)
    elif(f == 0):
        return 1
    else:
        return 0


def f2(x, f):		
    if(f == 2):
        return ((x-1970.)/100.)**2.
    elif(f == 1):
        return ((x-1970.)/100.)
    elif(f == 0):
        return np.cos((x-1970)/100)
    else:
        return 0

#Belegung der Daten
x = dat('data')[0]
y = dat('data')[1]
n = len(x)
 
 
#Berechnung der Koefizienten
a_max = 3
A = np.empty([n,a_max])

for i in range(a_max):
	A[:,i] = np.vectorize(f1)(x, i)
	
	
A = np.matrix(A)
y = np.matrix(y)
a = np.linalg.solve(A.T*A,A.T*y.T)
print 'Die optimalen Paramter für die (i) lauten:'
print a

b_max = 3
B = np.empty([n,b_max])
for i in range(a_max):
    B[:,i] = np.vectorize(f2)(x, i)

B = np.matrix(B)
b = np.linalg.solve(B.T*B,B.T*y.T)
print 'Die optimalen Paramter für die (ii) lauten:'
print b


#Definition der Fit-Funktionen 
def F1(x):
	y = 0
	for i in range(len(a)):
		y += a[i]*f1(x,i)
	return y

def F2(x):
	y = 0
	for i in range(len(b)):
		y += b[i]*f2(x,i)
	return y

	
x = dat('data')[0]
y = dat('data')[1]	
sum = 0


#Fehlerrechnung
for i in range(n):
	sum +=(y[i]-F1(x[i]))**2

print 'Summe der Fehlerquadrate für (i):' 
print sum

for i in range(n):
	sum +=(y[i]-F2(x[i]))**2
print 'Summe der Fehlerquadrate für (ii):'
print sum
       

max = 0
for i in range(n):
	check = (y[i]-F1(x[i]))**2
	if check > max:
		max = check
		index = i
print 'Maximaler Fehler für (i):'
print max
print 'Dieser tritt im folgenden Jahr auf::'
print x[index]

max = 0
for i in range(n):
	check = (y[i]-F2(x[i]))**2
	if check > max:
		max = check
		index = i
print 'Maximaler Fehler für (ii):'
print max
print 'Dieser tritt im folgenden Jahr auf::'
print x[index]
	   

#Plot, wurde als figure_1.png exportiert.
#x = np.arange(1970.,2030.,1.)
#plt.plot(dat('data')[0], dat('data')[1], 'bs', x, np.vectorize(F1)(x), x, np.vectorize(F2)(x))
#plt.legend(('Daten', '$(i)$', '$(ii)$'), loc=1)
#plt.xlabel('Jahr')
#plt.ylabel('Doenerpreis in Euro')
#plt.show()
print 'Im Jahr 2020 kostet ein Döner nach Ausgleichsunfktion (i):'
print F1(2020)
print 'Im Jahr 2020 kostet ein Döner nach Ausgleichsunfktion (ii):'
print F2(2020)




#6.2
#Funktionen, partielle Ableitungen und Startparameter
def f(a, t):
    return np.exp(-a[0]*t)*(a[2]*np.sin(a[1]*t)+a[3]*np.cos(a[1]*t))

def fa(i, a, t):
    if(i == 0):
        return -t*np.exp(-a[0]*t)*(a[2]*np.sin(a[1]*t)+a[3]*np.cos(a[1]*t))
    elif(i == 1):
        return t*np.exp(-a[0]*t)*(a[2]*np.cos(a[1]*t)-a[3]*np.sin(a[1]*t))
    elif(i == 2):
        return np.exp(-a[0]*t)*np.sin(a[1]*t)
    elif(i == 3):
        return np.exp(-a[0]*t)*np.cos(a[1]*t)
    else:
        return 0

def g(a,t):
	return np.exp(-a[0]*t)*a[2]*np.sin(a[1]*t)

def ga(i, a, t):
    if(i == 0):
        return -t*np.exp(-a[0]*t)*a[2]*np.sin(a[1]*t)
    elif(i == 1):
        return np.exp(-a[0]*t)*a[2]*t*np.cos(a[1]*t)
    elif(i == 2):
        return np.exp(-a[0]*t)*np.sin(a[1]*t)
    else:
        return 0	
	

a1 = np.array([0.8, 6.4, 4.2,-0.3])
a2 = np.array([0.3, 5.4, 7.2,-1.3])
a3 = np.array([1.0, 7.0,-6.0, 3.0])
b1 = np.array([0.8, 6.4, 4.2])
b2 = np.array([0.3, 5.4, 7.2])
b3 = np.array([1.0, 7.0,-6.0])	
	




t = dat('data2', " ", 0)[0]
y = dat('data2', " ", 0)[1]
n = len(t)



#Implementierung Gauß-Newton mit Dämpfung:


def jac(funca, n, m, a, t):
    D = np.empty([n, m])
    for i in range(m):
        D[:,i] = np.vectorize(funca, excluded=[0, 1])(i, a, t)
    return np.matrix(D)

def error(func, a, t, y):
    return np.linalg.norm(np.vectorize(func, excluded=[0])(a, t) - y)

def damp(func, a0, t, y, delta):
    damp = 1.
    err = error(func, a0, t, y)
    while(damp > 1e-6):
        a = a0 + damp * delta
        if(err > error(func, a, t, y)):
            break
        damp /= 2
    return damp


#iterate
def iterate(func, funca, n, m, a0, t, stop = 1e-6, damping = False):
    a = a0
    while(True):
        a_old = a
        g = np.matrix(y - np.vectorize(func, excluded=[0])(a, t))

        D = jac(funca, n, m, a, t)
        delta = np.squeeze(np.asarray(np.linalg.solve(D.T*D, D.T*g.T)))


        if(damping):
            delta *= damp(func, a, t, y, delta)
        a = a + delta

        #print np.linalg.norm(a-a_old)
        if(np.linalg.norm(a-a_old) < stop):
            return a



			
			
#Lösungsvektoren
print 'Paramter für a) ohne Dämpfung'			
sola1 = iterate(f, fa, n, 4, a1, t, 1e-6)
sola2 = iterate(f, fa, n, 4, a2, t, 1e-6)
sola3 = iterate(f, fa, n, 4, a3, t, 1e-6)
print sola1
print sola2
print sola3

print 'Paramter für b) ohne Dämpfung'
solb1 = iterate(g, ga, n, 3, b1, t, 1e-6)
solb2 = iterate(g, ga, n, 3, b1, t, 1e-6)
solb3 = iterate(g, ga, n, 3, b1, t, 1e-6)
print solb1
print solb2
print solb3

print 'Paramter für a) mit Dämpfung'			
sola1 = iterate(f, fa, n, 4, a1, t, 1e-6,True)
sola2 = iterate(f, fa, n, 4, a2, t, 1e-6,True)
sola3 = iterate(f, fa, n, 4, a3, t, 1e-6,True)
print sola1
print sola2
print sola3

print 'Paramter für b) mit Dämpfung'
solb1 = iterate(g, ga, n, 3, b1, t, 1e-6,True)
solb2 = iterate(g, ga, n, 3, b1, t, 1e-6,True)
solb3 = iterate(g, ga, n, 3, b1, t, 1e-6,True)
print solb1
print solb2
print solb3						
												
#Ohne Dämpfung ergeben sich für den Startwert a3) keine sinnvollen Werte. 



#Test-Plot:		
#x = np.arange(1.,5.,0.001)
#plt.plot(x,g(sola1,x),dat('data2', " ", 0)[0],dat('data2', " ", 0)[1],'x')
#plt.legend(('Fit','Datensatz'), loc=1)
#plt.xlabel('x')
#plt.ylabel('t')
#plt.show()

			


#6.4.2												
#Der Plot wurde als figure_2.png exportiert. Für das letzte Viertel der Messwerte ergab sich
#ein Overflow in der Exponentialfunktion von f(a,t).

#Plot
x = np.arange(1.,5.,0.001)

#erstes viertel:
t = np.split(dat('data2', " ", 0)[0],4)[0]
y = np.split(dat('data2', " ", 0)[1],4)[0]
n = len(t)
sol = iterate(f, fa, n, 4, a1, t, 1e-2)
x = np.arange(1.,5.,0.001)
plt.plot(x,g(sol,x))


#letztes viertel
#t = np.split(dat('data2', " ", 0)[0],4)[3]
#y = np.split(dat('data2', " ", 0)[1],4)[3]
#n = len(t)
#sol = iterate(f, fa, n, 4, a1, t, 1e-2)
#x = np.arange(1.,5.,0.001)
#plt.plot(x,g(sol,x))





#Hilfsfunktion für jeden k-ten Wert der Liste f
def teilfunk(f,k):
	na = np.array([])
	c = 0
	for i in range(0,len(f)):
		c+=1
		if c == k:
			na = np.append(na,np.array(f[i]))
			c=0
	return na

#jeder fünfte

t = teilfunk(dat('data2', " ", 0)[0],5)
y = teilfunk(dat('data2', " ", 0)[1],5)
n = len(t)
sol = iterate(f, fa, n, 4, a1, t, 1e-8)
x = np.arange(1.,5.,0.001)
plt.plot(x,g(sol,x))



#jeder zwanzigste
t = teilfunk(dat('data2', " ", 0)[0],20)
y = teilfunk(dat('data2', " ", 0)[1],20)
n = len(t)
sol = iterate(f, fa, n, 4, a1, t, 1e-8)
x = np.arange(1.,5.,0.001)
plt.plot(x,g(sol,x))


#jeder vierzigste
t = teilfunk(dat('data2', " ", 0)[0],40)
y = teilfunk(dat('data2', " ", 0)[1],40)		
n = len(t)
sol = iterate(f, fa, n, 4, a1, t, 1e-8)
x = np.arange(1.,5.,0.001)
plt.plot(x,g(sol,x))

plt.plot(dat('data2', " ", 0)[0],dat('data2', " ", 0)[1],'x')
plt.xlabel('x')
plt.ylabel('t')
plt.legend(('1. Viertel','5. Wert','20. Wert','40. Wert','Datensatz'), loc=1)
#plt.show()


