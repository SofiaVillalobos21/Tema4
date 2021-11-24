
# Para manipular imágenes (Python Imaging Library)
from PIL import Image

# Para manipular 'arrays' de pixeles y bits, señales y operaciones
import numpy as np

# Para visualizar imágenes y señales
import matplotlib.pyplot as plt

# Para medir el tiempo de simulación
import time

def fuente_info(imagen):
    '''Una función que simula una fuente de
    información al importar una imagen y 
    retornar un vector de NumPy con las 
    dimensiones de la imagen, incluidos los
    canales RGB: alto x largo x 3 canales

    :param imagen: Una imagen en formato JPG
    :return: un vector de pixeles
    '''
    img = Image.open(imagen)
    
    return np.array(img)

def rgb_a_bit(array_imagen):
    '''Convierte los pixeles de base 
    decimal (de 0 a 255) a binaria 
    (de 00000000 a 11111111).

    :param imagen: array de una imagen 
    :return: Un vector de (1 x k) bits 'int'
    '''
    # Obtener las dimensiones de la imagen
    x, y, z = array_imagen.shape
    
    # Número total de elementos (pixeles x canales)
    n_elementos = x * y * z

    # Convertir la imagen a un vector unidimensional de n_elementos
    pixeles = np.reshape(array_imagen, n_elementos)

    # Convertir los canales a base 2
    bits = [format(pixel, '08b') for pixel in pixeles]
    bits_Rx = np.array(list(''.join(bits)))
    
    return bits_Rx.astype(int)


def canal_ruidoso(senal_Tx, Pm, SNR):
    '''Un bloque que simula un medio de trans-
    misión no ideal (ruidoso) empleando ruido
    AWGN. Pide por parámetro un vector con la
    señal provieniente de un modulador y un
    valor en decibelios para la relación señal
    a ruido.

    :param senal_Tx: El vector del modulador
    :param Pm: Potencia de la señal modulada
    :param SNR: Relación señal-a-ruido en dB
    :return: La señal modulada al dejar el canal
    '''
    # Potencia del ruido generado por el canal
    Pn = Pm / pow(10, SNR/10)

    # Generando ruido auditivo blanco gaussiano (potencia = varianza)
    ruido = np.random.normal(0, np.sqrt(Pn), senal_Tx.shape)

    # Señal distorsionada por el canal ruidoso
    senal_Rx = senal_Tx + ruido

    return senal_Rx

def bits_a_rgb(bits_Rx, dimensiones):
    '''Un blque que decodifica el los bits
    recuperados en el proceso de demodulación

    :param: Un vector de bits 1 x k 
    :param dimensiones: Tupla con dimensiones de la img.
    :return: Un array con los pixeles reconstruidos
    '''
    # Cantidad de bits
    N = len(bits_Rx)

    # Se reconstruyen los canales RGB
    bits = np.split(bits_Rx, N / 8)

    # Se decofican los canales:
    canales = [int(''.join(map(str, canal)), 2) for canal in bits]
    pixeles = np.reshape(canales, dimensiones)

    return pixeles.astype(np.uint8)

# 4. - Asignaciones del proyecto

# 4.1. - Modulación 8-PSK

# 4.1.1 Se establecen las funciones necesarias**

import numpy as np


def modulador_8_PSK(bits, fc, mpp):
    '''Un método que simula el esquema de modulación digital 8-PSK.

    :param bits: Vector unidimensional de bits
    :param fc: Frecuencia de la portadora en Hz
    :param mpp: Cantidad de muestras por periodo de onda portadora
    :return: Un vector con la señal modulada
    :return: Un valor con la potencia promedio [W]
    :return: La onda portadora1 c1(t) = cos(2πfct)
    :return: La onda portadora2 c2(t) = sin(2πfct)
    '''
    # 1. Parámetros de la 'señal' de información (bits)
    N = len(bits)  # Cantidad de bits

    # 2. Construyendo un periodo de la señal portadora c(t)
    Tc = 1 / fc  # periodo [s]
    t_periodo = np.linspace(0, Tc, mpp)  # mpp: muestras por período
    portadora1 = np.cos(2*np.pi*fc*t_periodo)  # cos(2πfct)
    portadora2 = np.sin(2*np.pi*fc*t_periodo)  # sin(2πfct)

    # 3. Inicializar la señal modulada s(t)
    t_simulacion = np.linspace(0, N*Tc, N*mpp)
    senal_Tx = np.zeros(t_simulacion.shape)

    # 4. Asignar las formas de onda según los bits (8-PSK)
    h = np.sqrt(2)/2
    for i in range(0, N, 3):
        if bits[i] == 1 and bits[i+1] == 1 and bits[i+2] == 1:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * 1 + portadora2 * 0

        elif bits[i] == 1 and bits[i+1] == 1 and bits[i+2] == 0:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * h + portadora2 * h

        elif bits[i] == 0 and bits[i+1] == 1 and bits[i+2] == 0:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * 0 + portadora2 * 1

        elif bits[i] == 0 and bits[i+1] == 1 and bits[i+2] == 1:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * -h + portadora2 * h

        elif bits[i] == 0 and bits[i+1] == 0 and bits[i+2] == 1:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * -1 + portadora2 * 0

        elif bits[i] == 0 and bits[i+1] == 0 and bits[i+2] == 0:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * -h + portadora2 * -h

        elif bits[i] == 1 and bits[i+1] == 0 and bits[i+2] == 0:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * 0 + portadora2 * -1

        else:
            senal_Tx[i*mpp: (i+1)*mpp] = portadora1 * h + portadora2 * -h

    # 5. Calcular la potencia promedio de la señal modulada
    Pm = (1 / (N*Tc)) * np.trapz(pow(senal_Tx, 2), t_simulacion)

    return senal_Tx, Pm, portadora1, portadora2


def demodulador_8_PSK(senal_Rx, portadora1, portadora2, mpp):
    '''Un método que simula un bloque demodulador
    de señales, bajo un esquema BPSK. El criterio
    de demodulación se basa en decodificación por
    detección de energía.

    :param senal_Rx: La señal recibida del canal
    :param portadora: La onda portadora c(t)
    :param mpp: Número de muestras por periodo
    :return: Los bits de la señal demodulada
    '''
    # Cantidad de muestras en senal_Rx
    M = len(senal_Rx)

    # Cantidad de bits (símbolos) en transmisión
    N = int(M / mpp)

    # Vector para bits obtenidos por la demodulación
    bits_Rx = np.zeros(N)

    # Vector para la señal demodulada
    senal_demodulada = np.zeros(senal_Rx.shape)

    # Pseudo-energía de un período de la portadora1
    Es1 = np.sum(portadora1 * portadora1)

    # Pseudo-energía de un período de la portadora2
    Es2 = np.sum(portadora2 * portadora2)

    h = np.sqrt(2)/2

    # Demodulación
    for i in range(N):
        # Producto interno de dos funciones
        producto1 = senal_Rx[i*mpp: (i+1)*mpp] * portadora1
        Ep1 = np.sum(producto1)
        producto2 = senal_Rx[i*mpp: (i+1)*mpp] * portadora2
        Ep2 = np.sum(producto2)
        senal_demodulada[i*mpp: (i+1)*mpp] = producto1 + producto2

        # Criterio de decisión por detección de energía
        if i % 3 == 0:
            if (Ep1 >= (1+h)/2*Es1 and -h/2*Es2 <= Ep2 <= h/2*Es2):
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
            elif (h/2*Es1 <= Ep1 <= (1+h)/2*Es1 and
                  h/2*Es2 <= Ep2 <= (1+h)/2*Es2):
                bits_Rx[i] = 1
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
            elif (-h/2*Es1 <= Ep1 <= h/2*Es1 and Ep2 >= (1+h)/2*Es2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 0
            elif (-(1+h)/2*Es1 <= Ep1 <= -h/2*Es1 and
                  h/2*Es2 <= Ep2 <= (1+h)/2*Es2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 1
                bits_Rx[i+2] = 1
            elif (Ep1 <= -(1+h)/2*Es1 and -h/2*Es1 <= Ep2 <= h/2*Es2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1
            elif (-(1+h)/2*Es1 <= Ep1 <= -h/2*Es1 and
                  -(1+h)/2*Es2 <= Ep2 <= -h/2*Es2):
                bits_Rx[i] = 0
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
            elif (-h/2*Es1 <= Ep1 <= h/2*Es1 and Ep2 <= -(1+h)/2*Es2):
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 0
            else:
                bits_Rx[i] = 1
                bits_Rx[i+1] = 0
                bits_Rx[i+2] = 1

    return bits_Rx.astype(int), senal_demodulada

import numpy as np
import matplotlib.pyplot as plt
import time


def Modulación_8PSK(SNR):
    # Parámetros
    fc = 5000  # frecuencia de la portadora
    mpp = 20   # muestras por periodo de la portadora

    # Iniciar medición del tiempo de simulación
    inicio = time.time()

    # 1. Importar y convertir la imagen a trasmitir
    imagen_Tx = fuente_info(requests.get(
        'https://github.com/DavidMairena/Tema4/blob/main/arenal.jpg?raw=true',
        stream=True).raw)
    dimensiones = imagen_Tx.shape

    # 2. Codificar los pixeles de la imagen
    bits_Tx = rgb_a_bit(imagen_Tx)

    # 3. Modular la cadena de bits usando el esquema BPSK
    senal_Tx, Pm, portadora1, portadora2 = modulador_8_PSK(bits_Tx, fc, mpp)

    # 4. Se transmite la señal modulada, por un canal ruidoso
    senal_Rx = canal_ruidoso(senal_Tx, Pm, SNR)

    # 5. Se desmodula la señal recibida del canal
    bits_Rx, senal_demodulada = demodulador_8_PSK(senal_Rx, portadora1,
                                                  portadora2, mpp)

    # 6. Se visualiza la imagen recibida
    imagen_Rx = bits_a_rgb(bits_Rx, dimensiones)
    Fig = plt.figure(figsize=(10, 6))

    # Cálculo del tiempo de simulación
    print('Duración de la simulación: ', time.time() - inicio)

    # 7. Calcular número de errores
    errores = sum(abs(bits_Tx - bits_Rx))
    BER = errores/len(bits_Tx)
    print('{} errores, para un BER de {:0.4f}.'.format(errores, BER))

    # Mostrar imagen transmitida
    ax = Fig.add_subplot(1, 2, 1)
    imgplot = plt.imshow(imagen_Tx)
    ax.set_title('Transmitido')

    # Mostrar imagen recuperada
    ax = Fig.add_subplot(1, 2, 2)
    imgplot = plt.imshow(imagen_Rx)
    ax.set_title('Recuperado')
    Fig.tight_layout()

    plt.imshow(imagen_Rx)

    # Visualizar el cambio entre las señales
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, sharex=True, figsize=(14, 7))

    # La señal modulada por (8-PSK)
    ax1.plot(senal_Tx[0:600], color='g', lw=2)
    ax1.set_ylabel('$s(t)$')

    # La señal modulada al dejar el canal
    ax2.plot(senal_Rx[0:600], color='b', lw=2)
    ax2.set_ylabel('$s(t) + n(t)$')

    # La señal demodulada
    ax3.plot(senal_demodulada[0:600], color='m', lw=2)
    ax3.set_ylabel('$b^{\prime}(t)$')
    ax3.set_xlabel('$t$ / milisegundos')
    fig.tight_layout()
    plt.show()

    return senal_Tx


# 4.1.3 Simulación para la modulación 8-PSK:
# Se realizan la prueba con diferentes relaciones señal-a-ruido del canal (SNR)

SNR1 = -5   # Mayor ruido que la señal, va a tener muchos errores
senal_Tx1 = Modulación_8PSK(SNR1)

SNR2 = 5   # Una proporción entre la señal y el ruido
senal_Tx2 = Modulación_8PSK(SNR2)

SNR3 = 15   # La señal es mucho clara que el ruido
# En este caso si la demodulación está bien hecha no debería haber ningún error
senal_Tx = Modulación_8PSK(SNR3)

### 4.2. - Estacionaridad y ergodicidad

import numpy as np
import matplotlib.pyplot as plt

# Tiempo de muestreo
t_x = np.linspace(0, 0.01, 100)
R = [1, -1]

# Formas de la onda:
X_t = np.empty((4, len(t_x)))

# Nueva figura
plt.figure()

# Matriz con los posibles valores de cada función
for i in R:
    x1 = i * np.cos(2 * (np.pi) * fc * t_x) + i * np.sin(2 * (np.pi) * fc *
                                                         t_x)
    x2 = -i * np.cos(2 * (np.pi) * fc * t_x) + i * np.sin(2 * (np.pi) * fc *
                                                          t_x)
    X_t[i, :] = x1
    X_t[i+1, :] = x2
    plt.plot(t_x, x1, lw=4, color='g')
    plt.plot(t_x, x2, lw=4, color='c')

# Se determina un promedio de las 4 realizaciones para cada instante:
PR = [np.mean(X_t[:, i]) for i in range(len(t_x))]
plt.plot(t_x, PR, lw=6, color='k', label='Promedio de realizaciones')

# Se grafica el resultado teórico del valor esperado:
S = senal_Tx.astype('float')
RT = np.mean(S) * t_x
plt.plot(t_x, RT, '-.', lw=3, color='y', label='Valor teórico esperado')

# Se muestran las realizaciones, junto al promedio calculado y teórico:

plt.title('Realizaciones del proceso aleatorio $X(t)$')
plt.xlabel('$t$')
plt.ylabel('$x_i(t)$')
plt.legend()
plt.show()



# Cuando hablamos de ergodicidad, estamos hablando de un efecto que ocurre en algunos sistemas mecánicos en donde

### 4.3. - Densidad espectral de potencia

from scipy import fft

# Transformada de Fourier
senal_f = fft(senal_Tx)

# Muestras de la señal
Nm = len(senal_Tx)

# Número de símbolos (198 x 89 x 8 x 3)
Ns = Nm // mpp

# Tiempo del símbolo = periodo de la onda portadora
Tc = 1 / fc

# Tiempo entre muestras (período de muestreo)
Tm = Tc / mpp

# Tiempo de la simulación
T = Ns * Tc

# Espacio de frecuencias
f = np.linspace(0.0, 1.0/(2.0*Tm), Nm//2)

# Gráfica
plt.figure(figsize=(18,10))
plt.plot(f, 2.0/Nm * np.power(np.abs(senal_f[0:Nm//2]), 2))
plt.xlim(0, 20000)
plt.grid()
plt.show()
