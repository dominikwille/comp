#!/usr/local/bin/python
# -*- coding: utf-8 -*-
#
# @author Dominik Wille
# @author Stefan Pojtinger
# @tutor Alexander Schlaich
# @sheet 10
#
#Packete einlesen:
import numpy as np
import matplotlib.pyplot as plt
from scipy import special





#Aufgabe 11.2

def f(x):
	return np.sin(x)+2*np.sin(0.6*x)+0.5*np.sin(4*x)

	
def absF(N,f,w):
	return np.abs(np.fft.fft(f(np.arange(0, N, 0.05))*w(N,np.arange(0, N, 0.05))))**2
	

def absFplt(N,f,w):
	return plt.plot(np.arange(0, N, 0.05),absF(N,f,w))

def w(N,i):
	return 1

def w1(N,i):
	if i <= N/10:
		return 10/N*i
	
	elif i >= N-N/10:
		return 10/N * (N-i)
	
	else:
		return 1
		
def w2(N,i):
	return 0.42 - 0.5 * np.cos(2*np.pi*i/N) + 0.08*np.cos(4*np.pi*i/N)

vw1 = np.vectorize(w1)
	
	
	
#vecw1 = np.vectorize(w1)
#x = np.arange(0, 20, 0.05)
#y = vecw1(20.,x)
#plt.plot(x,y)
#absFplt(2000,f,w)
#absFplt(10000,f,w)
#absFplt(2000,f,w2)
#absFplt(10000,f,w2)
#absFplt(2000,f,vw1)
#absFplt(10000,f,vw1)
#plt.yscale('log')
#plt.legend(('Ohne Fensterfunktion N=2000', 'Ohne Fensterfunktion N=10000', 'Trapezfenster N=2000', 'Trapezfenster N=10000', 'Blackman-Fenster N=2000', 'Blackman-Fenster N=10000'), loc=0)
#plt.show()