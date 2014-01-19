#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 11
#
#Packete einlesen:
import numpy as np
import matplotlib.pyplot as plt





#Aufgabe 11.2

def f(x):
	return np.sin(x)+2*np.sin(0.6*x)+0.5*np.sin(4*x)

	
def absF(N,f,w):
	return np.abs(np.fft.fft(np.append(f(np.arange(0, N, 0.05)),(np.zeros(N/0.05))))*w(N,np.arange(0, 2*N, 0.05)))**2
	

	
def absFplt(N,f,w):
	return plt.plot(np.arange(0, N, 0.05),absF(N,f,w)[:-N/0.05])

def w(N,i):
	return 1

def w1(N,i):
	
	if i >= N:
		return 0
		
	elif i < 0:
		return 0	
	
	elif i <= N/10.:
		return 10./N*i
	
	elif i >= N-N/10:
		return 10./N * (N-i)
	
	else:
		return 1
	
		
def w2(N,i):
	return 0.42 - 0.5 * np.cos(2*np.pi*i/N) + 0.08*np.cos(4*np.pi*i/N)

vw1 = np.vectorize(w1)
	



#absFplt(2000.,f,w)
#absFplt(10000.,f,w)
#absFplt(2000.,f,w2)
#absFplt(10000.,f,w2)
#absFplt(2000.,f,vw1)
#absFplt(10000.,f,vw1)
#plt.xscale('log')
#plt.legend(('Ohne Fensterfunktion N=2000', 'Ohne Fensterfunktion N=10000', 'Trapezfenster N=2000', 'Trapezfenster N=10000', 'Blackman-Fenster N=2000', 'Blackman-Fenster N=10000'), loc=0)
#plt.show()



#Aufgabe 11.3

def s(t):
	return np.exp( -5*(t-1.5)**2 ) + 0.5 * np.exp( -2*(t-3)**4 )


def signalgen(f, tmin, tmax, dt):
	c = []
	n = []
	t = tmin
	while t < tmax-dt:
		r = np.random.random()
		c.append(s(t)+r)
		n.append(r)
		t += dt
	return c,n
	
signal = np.copy(signalgen(s, 0, 5, 0.02))


#x = np.arange(0, 5, 0.02)
#plt.plot(x,signal[0])
#plt.xlabel('t')
#plt.ylabel('c(t)')
#plt.show()


#absC = np.abs(np.fft.fft(np.append(signal[0],np.zeros(5./0.02))))**2
#absN = np.abs(np.fft.fft(np.append(signal[1],np.zeros(5./0.02))))**2
#x = np.arange(0, 5, 0.02)
#plt.plot(x,absC[:-5./0.02],x,absC[:-5./0.02]-absN[:-5./0.02])
#plt.xlabel('t')
#plt.ylabel('C(t)')
#plt.legend(('mit Rauschen', 'ohne Rauschen'), loc=1)
#plt.show()



