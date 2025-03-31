import numpy as np
import matplotlib.pyplot as plt
import signals as sigs
from scipy.fftpack import fft
from scipy import signal

# transformación fft para detectar la frecuencia espectral de cada señal de la serie temporal
freq_domain_signal = fft(sigs.cardiaca_100Hz)
plt.stem(freq_domain_signal)
plt.xlim(0,100)
plt.title('Analisis de frecuencia de entrada') 
plt.show()

######filtros FIR
#vector de analisis normalizado
signal_ecg=np.zeros(len(sigs.cardiaca_100Hz))

#normalizar la señal de entrada
for i,num in enumerate (sigs.cardiaca_100Hz):
    signal_ecg[i]=float(sigs.cardiaca_100Hz[i]/max(sigs.cardiaca_100Hz))
    
#grafica normalizada
plt.plot(signal_ecg)
plt.title('Señal normalizada') 
plt.show()    
    
#generación de filtro por ventana
###############################################################################
#filtro pasa banda
bandpas_coef=signal.firwin(91,[2,30],nyq=100,pass_zero=False,window='blackman')

#grafica de coeficientes
plt.plot(bandpas_coef)
plt.title('Coeficientes pasabandas') 
plt.show()
###############################################################################
#convolucion
signal_output=signal.convolve(signal_ecg, bandpas_coef,mode='same')

#grafica de resultados
plt.plot(signal_output, color='red')
plt.plot(signal_ecg, color='blue')
plt.title('Entrada (Azul) vs Salida (Rojo)')
plt.show()
###############################################################################
#analisis de frecuencia
freq_domain_signal = fft(signal_output)
plt.stem(signal_output)
plt.title('Analisis de frecuencia de salida') 
plt.xlim(0,100)
plt.show()

#Modulo de la señal
modulo = np.abs(signal_output)
plt.plot(modulo)
plt.title('Modulo')
plt.show()

#Fase de la señal
fase = np.angle(signal_output, deg=True)
plt.plot(fase)
plt.title('Fase')
plt.show()