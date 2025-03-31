import numpy as np
import matplotlib.pylab as plt
import padasip as pa

# Creacion de los datos
N = 500
x = np.random.normal(0, 1, (N, 4)) # matriz de entrada
v = np.random.normal(0, 0.1, N) # ruido
d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # destino

# Identificacion
f = pa.filters.FilterLMS(n=4, mu=0.1, w="random")
y, e, w = f.run(d, x)

# Imprimo los resultados
plt.figure(figsize=(15,9))
plt.subplot(211);plt.title("Adecuacion");plt.xlabel("muestras - k")
plt.plot(d,"b", label="d - destino")
plt.plot(y,"g", label="y - salida");plt.legend()
plt.subplot(212);plt.title("Error del filtro");plt.xlabel("muestras - k")
plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
plt.tight_layout()
plt.show()

#Modulo de la señal
modulo = abs(10*np.log10(e**2))
plt.plot(t,modulo)
plt.title('Modulo')
plt.show()

#Fase de la señal
fase = np.angle(10*np.log10(e**2))
plt.plot(t,fase)
plt.title('Fase')
plt.show()