import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# Definimos las frecuencias de corte baja y alta
f1, f2 = 300, 3000  
# Convertimos las frecuencias a radianes por segundo
wn1, wn2 = f1 * 2 * np.pi, f2 * 2 * np.pi  

# Definimos el orden del filtro
N = 2

# Creamos el filtro pasabanda utilizando la función iirfilter
b2, a2 = signal.iirfilter(N, [wn1, wn2], btype='band', analog=True, ftype='butter')

# Generamos un array de frecuencias espaciadas logarítmicamente
frequencies = np.logspace(np.log10(30), np.log10(30000), num=1000)

# Calculamos la respuesta del filtro en las frecuencias especificadas
ws2, hs2 = signal.freqs(b2, a2, worN=frequencies*2*np.pi)  
# Convertimos las frecuencias a Hz
wsHz2 = ws2 / (2 * np.pi)

# Convertimos la magnitud a dB
hs2_dB = 20 * np.log10(np.abs(hs2))
# Convertimos la fase a grados
hs2_phase = np.angle(hs2, deg=True)

# Creamos una nueva figura y obtenemos el objeto de los ejes
fig, ax1 = plt.subplots(figsize=(14,8))  

color = 'tab:blue'
# Configuramos los ejes y etiquetas
ax1.set_xlabel('Frequencia [Hz]')
ax1.set_ylabel('Magnitud [dB]', color=color)  
# Graficamos la magnitud en dB
ax1.plot(wsHz2, hs2_dB, color=color)  
ax1.tick_params(axis='y', labelcolor=color)
# Configuramos el eje x para que sea logarítmico
plt.xscale('log')  
# Establecemos los límites del eje x
ax1.set_xlim([30, 30000])  
ax1.set_ylim([-50, 1])
# Configuramos las líneas de la cuadrícula
ax1.grid(which='both')  

# Creamos un segundo eje que comparte el mismo eje x
ax2 = ax1.twinx()  
color = 'tab:red'
# Configuramos los ejes y etiquetas para la fase
ax2.set_ylabel('Fase [grados]', color=color)  
# Graficamos la fase en grados
ax2.plot(wsHz2, hs2_phase, color=color)  
ax2.tick_params(axis='y', labelcolor=color)

# Ajustamos el diseño para evitar la superposición
fig.tight_layout()  
# Mostramos el gráfico de la transferencia y la fase
plt.show()

####################################################################################################

# Definimos las frecuencias de corte baja y alta para la señal de entrada
f1, f2 = 20, 20000
# Cantidad de componentes de frecuencia
n = 50

# Generamos la señal de entrada
t = np.linspace(0, 1, 50000, False)  # 1 segundo
frequencies = np.random.uniform(f1, f2, n)  # Frecuencias aleatorias entre f1 y f2
signal_input = np.sum([np.sin(2 * np.pi * f * t) for f in frequencies], axis=0)

# Calculamos la FFT de la señal de entrada
fft_result = np.fft.fft(signal_input)

# Calculamos las frecuencias correspondientes
frequencies = np.fft.fftfreq(len(signal_input), 1/50000)

# Convertimos el filtro analógico a formato zpk
z, p, k = signal.butter(N, [wn1, wn2], btype='band', analog=True, output='zpk')

# Convertimos el filtro analógico a digital
zd, pd, kd = signal.bilinear_zpk(z, p, k, fs=50000)

# Convertimos a formato SOS
sos = signal.zpk2sos(zd, pd, kd)

# Aplicamos el filtro digital
filtered = signal.sosfilt(sos, signal_input)

# Calculamos la FFT de la señal de salida
fft_result_output = np.fft.fft(filtered)

# Calculamos las frecuencias correspondientes
frequencies_output = np.fft.fftfreq(len(filtered), 1/50000)

# Creamos una figura para los gráficos
fig, axs = plt.subplots(2, 2, figsize=(14,8))

# Graficamos la señal de entrada en el tiempo
axs[0, 0].plot(t, signal_input)
axs[0, 0].set_title('Señal de entrada en el tiempo')
axs[0, 0].set_xlabel('Tiempo [s]')
axs[0, 0].set_ylabel('Amplitud')

# Graficamos las componentes de frecuencia de la señal de entrada
axs[0, 1].plot(np.abs(frequencies), np.abs(fft_result))

# Ajustamos el eje x dependiendo de donde estén las muestras
axs[0, 1].set_xlim([-100, f2 * 1.1])  # Mostramos hasta un poco más del máximo

axs[0, 1].set_title('Componentes de frecuencia de la señal de entrada')
axs[0, 1].set_xlabel('Frecuencia [Hz]')
axs[0, 1].set_ylabel('Amplitud')

# Graficamos la señal de salida en el tiempo
axs[1, 0].plot(t, filtered)
axs[1, 0].set_title('Señal de salida en el tiempo')
axs[1, 0].set_xlabel('Tiempo [s]')
axs[1, 0].set_ylabel('Amplitud')

# Graficamos las componentes de frecuencia de la señal de salida
axs[1, 1].plot(np.abs(frequencies_output), np.abs(fft_result_output))

# Ajustamos el eje x dependiendo de donde estén las muestras
axs[1, 1].set_xlim([-100, f2 * 1.1])  # Mostramos hasta un poco más del máximo

axs[1, 1].set_title('Componentes de frecuencia de la señal de salida')
axs[1, 1].set_xlabel('Frecuencia [Hz]')
axs[1, 1].set_ylabel('Amplitud')

# Ajustamos el diseño para evitar la superposición
plt.tight_layout()
# Mostramos el gráfico
plt.show()