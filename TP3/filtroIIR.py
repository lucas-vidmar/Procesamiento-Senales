import cmath
from scipy import signal 
import numpy as np
from matplotlib import pyplot as plt
from scipy.fftpack  import fft, fftshift

#butterworth (expliado en tps anteriores)
b, a = signal.butter(4, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.plot(w, 20 * np.log10(abs(h)))
plt.xscale('log')
plt.title('Butterworth filter frequency response')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.show()

#cheby1 (expliado en tps anteriores)
b, a = signal.cheby1(4, 5, 100, 'low', analog=True)
w, h = signal.freqs(b, a)
plt.semilogx(w, 20 * np.log10(abs(h)))
plt.title('Chebyshev Type I frequency response (rp=5)')
plt.xlabel('Frequency [radians / second]')
plt.ylabel('Amplitude [dB]')
plt.margins(0, 0.1)
plt.grid(which='both', axis='both')
plt.axvline(100, color='green') # cutoff frequency
plt.axhline(-5, color='green') # rp
plt.show()

#filtros iir
from numpy import cos,sin,pi,arange,absolute
sampling_rate=100
nsamples=400
t=arange(nsamples)/sampling_rate
x1=cos(2*pi*0.5*t)
x2=0.2*sin(2*pi*15.3*t)
x3=0.1*sin(2*pi*23.45*t+0.8)
x=x1+x2+x3 #Contruyo señal de ejemplo
plt.plot(x1)
plt.title('entrada')
plt.show()

#Filtrar datos a lo largo de una dimensión utilizando secciones de segundo orden en cascada.
#Prueba con filtro Chebychev
sos = signal.cheby1(10, 1, 1, 'lp', fs=100, output='sos')
filtered1 = signal.sosfilt(sos, x)
plt.plot(t,filtered1)
plt.title('Salida de Señal con filtro Chebyshev')
plt.show()

#Filtrar datos a lo largo de una dimensión utilizando secciones de segundo orden en cascada.
#Prueba con filtro Butterworth
sos = signal.butter(4, 1, 'lp', fs=100,analog=False, output='sos')
filtered2 = signal.sosfilt(sos, x)
plt.plot(t,filtered2)
plt.title('Salida de Señal con filtro Butterworth')
plt.show()

#Modulo de la señal
modulo = abs(filtered2)
plt.plot(t,modulo)
plt.title('Modulo')
plt.show()

#Fase de la señal
fase = np.angle(filtered2)
plt.plot(t,fase)
plt.title('Fase')
plt.show()