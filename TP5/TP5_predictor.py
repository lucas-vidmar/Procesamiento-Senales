############################################################################################
#                                        SECCIÓN IMPORT                                    #
############################################################################################

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read, write
from scipy.signal import butter, sosfiltfilt, sosfreqz ##hubo que usar sos porque ba es inestable
import sys

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    
############################################################################################
#                                        SECCIÓN INPUT                                     #
############################################################################################

print("Cargando audio de control...")
#Defino array de entrada 
samp2_t, sr2_t = tf.audio.decode_wav(tf.io.read_file("sweep.wav"), desired_channels=1)
samp2_t = tf.squeeze(samp2_t, axis=-1)

sr2_t = tf.cast(sr2_t, dtype=tf.int64)

sr2 = sr2_t.numpy()
samp2 = samp2_t.numpy()

print(samp2)
print(len(samp2))
print(sr2)

#defino línea de tiempo
long2 = samp2.shape[0]/sr2 #samples sobre samples por segundo igual segundos
t2 = np.arange(0, long2, 1/sr2)

print("Longitud del archivo de audio:", long2)

#Realizo la transformada de Fourier de los samples de audio
print("Transformando por FFT...")
fft2 = np.real(np.fft.fft(samp2))
fft2_freqs = np.real(np.fft.fftfreq(len(samp2))*sr2)

print(fft2)
print("Long FFT2: ", len(fft2))

fft2 = np.array(fft2, dtype=float)
fft2 = np.expand_dims(fft2, axis=1)

############################################################################################
#                                        SECCIÓN IA                                     #
############################################################################################

modelo = tf.keras.models.load_model("modelo.keras")

print("Realizando predicción...")

resultado = modelo.predict(fft2)

resultado_tr = resultado.reshape(1,resultado.shape[0],1)

print("Predicho!")
print(resultado)
print(resultado.shape)

print("Reshape")
print(resultado_tr)
print(resultado_tr.shape)

print("Guardando resultado de la IA en WAV...")
res_t = np.real(np.fft.ifft(resultado))
write("filt_pred.wav", sr2, res_t)
############################################################################################
#                                        SECCIÓN GRÁFICOS                                  #
############################################################################################

print("Graficando...")


#Gráficos IA
fig_ia, h = plt.subplots(1, 2)

h[0].set_xlabel("# Epoca")
h[0].set_ylabel("Magnitud de pérdida")
h[0].plot(historial.history["loss"])

h[1].set_xlabel("Tiempo [s]")
h[1].set_ylabel("Magnitud")
h[1].plot(t, res_t)

#Muestro gráficos
plt.show()
