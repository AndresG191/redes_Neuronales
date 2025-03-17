import tensorflow as tf
import numpy as np
from numpy.f2py.crackfortran import verbose
import matplotlib.pyplot as plt

#Aprendizaje supervisado-------

#Declaracion el arreglo de numeros de las entradas celsius
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
#Datos fahrenheit
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype= float)

#Diseño de red neuronal, utilizamos keras
capa = tf.keras.layers.Dense(units=1, input_shape=[1]) #Capa densa las que tienen conexiones con cada neurona
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1), #Adams permite a la red como ajustar a la red los pesos y sesgos de manera eficiente para que aprenda
    loss='mean_squared_error'
)

print("Comenzando entrenamiento...")
historial = modelo.fit(celsius, fahrenheit, epochs=1000, verbose = False) #epochs para decirle cuantas vueltas quiero que lo intente y optimizar
print("Modelo entrenado con éxito!")


plt.xlabel("#Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history["loss"])

print("Realizando predicción...")
input_data = np.array([[100]])
resultado = modelo.predict(input_data)
print("El reslutado es" +str(resultado), "fahrenheit!")