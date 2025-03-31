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
print("Leyendo audio...")
#Defino array de entrada 
##sample_rate, samples = read("ex.wav")
##samples = samples[:, 0] #me quedo con el canal izquierdo únicamente
##print(samples)

samples_t, sample_rate_t = tf.audio.decode_wav(tf.io.read_file("noise.wav"), desired_channels=1)
samples_t = tf.squeeze(samples_t, axis=-1)

sample_rate_t = tf.cast(sample_rate_t, dtype=tf.int64)

sample_rate = sample_rate_t.numpy()
samples = samples_t.numpy()

print(samples)
print(len(samples))
print(sample_rate)

#defino línea de tiempo
long = samples.shape[0]/sample_rate #samples sobre samples por segundo igual segundos
t = np.arange(0, long, 1/sample_rate)

print("Longitud del archivo de audio:", long)

#Realizo la transformada de Fourier de los samples de audio
print("Transformando por FFT...")
fft = np.real(np.fft.fft(samples))
##fft_norm = fft / np.max(fft)
fft_freqs = np.real(np.fft.fftfreq(len(samples))*sample_rate)

############################################################################################
#                                        SECCIÓN LTI                                       #
############################################################################################

#Defino mi LTI, en este caso un HPF
def hpf(samp, sr, fc1, fc2):
    nyq = 0.5*sr
    wn = [fc1, fc2]
    sos = butter(2, wn, btype='band', analog=False, output='sos', fs=sr)
    y = sosfiltfilt(sos, samp)
    return y

###Aplico filtro
print("Filtrando...")
fil = hpf(samples, sample_rate, 250, 1000)
print(fil)

print("Transformando audio filtrado por FFT...")
fil_fft = np.real(np.fft.fft(fil))

##MISC##
print("Convirtiendo audio filtrado...")
write("filt.wav", sample_rate, fil)


############################################################################################
#                                        SECCIÓN IA                                        #
############################################################################################
print("Entrenando IA...")

fftarr = np.array(fft, dtype=float)
filfftarr = np.array(fil_fft, dtype=float)

fftarr = np.expand_dims(fftarr, axis=1)
fftarr = np.expand_dims(filfftarr, axis=1)


in_ia = tf.keras.layers.Input(shape=fftarr.shape)


out_ia = tf.keras.layers.Conv1D(filters = 1,
                                kernel_size = 2,
                                strides = 2,
                                padding = "same",
                                activation = "relu",
                                input_shape = fftarr.shape[1:])(in_ia)



modelo = tf.keras.models.Model(inputs=in_ia, outputs=out_ia)

print("Compilando...")
modelo.compile(optimizer = tf.keras.optimizers.Adam(0.1),
               loss = 'mean_squared_error',
               metrics=["mse"])

print("Stats...")
historial = modelo.fit(fftarr, filfftarr, epochs = 10, verbose = True)
print("Entrenado!")

modelo.save("modelo.keras")
############################################################################################
#                                        SECCIÓN GRÁFICOS                                  #
############################################################################################

print("Graficando...")
#Multiples gráficos
fig, g = plt.subplots(2, 2)

#Gráfico en tiempo pre-filtro
g[0, 0].plot(t, samples)
g[0, 0].set_title("Amplitud en el tiempo - Prefiltro")
g[0, 0].set_xlabel("Tiempo [s]")
g[0, 0].set_ylabel("Magnitud")

#Grafico en frecuencia pre-filtro
g[0, 1].plot(np.abs(fft_freqs), np.real(fft))
g[0, 1].set_title("Amplitud en frecuencia - Prefiltro")
g[0, 1].set_xlabel("Frecuencias [Hz]")
g[0, 1].set_ylabel("Magnitud")

#Gráfico en tiempo post-filtro
g[1, 0].plot(t, fil)
g[1, 0].set_title("Amplitud en el tiempo - Postfiltro")
g[1, 0].set_xlabel("Tiempo [s]")
g[1, 0].set_ylabel("Magnitud")

#Grafico en frecuencia post-filtro
g[1, 1].plot(fft_freqs, np.real(fil_fft))
g[1, 1].set_title("Amplitud en frecuencia - Postfiltro")
g[1, 1].set_xlabel("Frecuencias [Hz]")
g[1, 1].set_ylabel("Magnitud")

#Muestro gráficos
plt.show()
