# coding=utf-8
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt

"""
BackPropagation, algoritmo de entrenamiento de redes neuronales.
================================================================
Entrenamiento de XOR
================================================================
Melvin Juárez 15015-14

"""

#error_permitido: Variable global para indicar cual es el error
#                 maximo permitido del algoritmo BackPropagation.
error_permitido = 0.01

"""
Funcion para conocer el gradiente de las neuronas de la capa K
--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|          e            |   Recibe el vector de errores      |
|                       |   individuales de la RNA.          |
------------------------+-------------------------------------
|          t_           |   Recibe el valor flotante         |
|                       |   de la derivada de la funcion de  |
|                       |   transferencia.                   |
------------------------+-------------------------------------
|        retorno        |   La funcion gradiente, que es la  |
|                       |   sumatoria de los errores         |
|                       |   individuales, por -1 * la prima  |
|                       |   de la funcion de transferencia,  |
|                       |   un numero flotante, en una lista.|
--------------------------------------------------------------
"""
def gradiente(e, t_):
    return sum(e)*(-1)*(t_)

"""
Funcion de gradiente local, nos da a conocer la gradiente de
la neurona i de la capa k-n.

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|      gradiente_k      |   Recibe el valor del gradiente de |
|                       |   la capa K.                       |
------------------------+-------------------------------------
|           w           |   Peso con el que esta conectado   |
|                       |   la neurona evaluada.             |
------------------------+-------------------------------------
|          t_           |   Recibe el valor de la derivada   |
|                       |   de la funcion de transferencia.  |
------------------------+-------------------------------------
|        retorno        |   La funcion de gradiente local,   |
|                       |   que funciona con la gradiente de |
|                       |   la capa siguiente, multiplicado  |
|                       |   por el peso y la prima de t, un  |
|                       |   numero flotante, en una lista.   |
--------------------------------------------------------------
"""
def gradiente_local(gradiente_k, w, t_):
    return gradiente_k*w*t_

"""
Funcion de correccion de pesos, retorna el vector de pesos ya corregido.

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|           w           |   Vector de pesos actuales de la   |
|                       |   neurona.                         |
------------------------+-------------------------------------
|      delta_omega      |   Valor de la funcion Delta_Omega. |
------------------------+-------------------------------------
|        retorno        |   Vector de nuevos pesos. Suma de  |
|                       |   pesos actuales más Delta.        |
--------------------------------------------------------------
"""
def delta(w,delta_omega):
    return w.T + delta_omega

"""
Funcion de diferencia entre los pesos actuales y los finales,
Delta_Omega.

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|       gradiente       |   Recibe el valor del gradiente de |
|                       |   la neurona evaluada.             |
------------------------+-------------------------------------
|           p           |   Vector de entradas de la neurona |
|                       |   evaluada.                        |
------------------------+-------------------------------------
|           n           |   Constante de entrenamiento, por  |
|                       |   defecto es 0.3.                  |
------------------------+-------------------------------------
|        retorno        |   Vector de flotantes, estos       |
|                       |   valores se suman a los pesos     |
|                       |   actuales para obtener los nuevos |
|                       |   pesos.                           |
--------------------------------------------------------------
"""
def delta_omega_min(gradiente,p,n = 0.3):
    return -n*gradiente*p

"""
Funcion de transferencia de la neurona,
recibe como parametro la energia (E) de la neurona.

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|           E           |   Energía de la neurona, un        |
|                       |   flotante.                        |
------------------------+-------------------------------------
|        retorno        |   Un flotante que indica el valor  |
|                       |   de salida de la neurona evaluada.|
--------------------------------------------------------------
"""
def transferencia(E):
    return 1 / (1 + np.exp(-E))

"""
Derivada de la funcion de transferencia, sirve para
el entrenamiento, recibe la energia (E) de la neurona.
En este caso es la derivada minificada, que es el valor
de la funcion menos su cuadrado.

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|           h           |   Es el flotante de salida de la   |
|                       |   neurona evaluada.                |
------------------------+-------------------------------------
|        retorno        |   El flotante de la derivada de la |
|                       |   funcion de transferencia de la   |
|                       |   neurona evaluada.                |
--------------------------------------------------------------

"""
def transferencia_min(h):
    return h - h**2


"""
Media cuadratica de los errores de la Red Neuronal

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|           e           |   Vector de errores individuales   |
|                       |   de cada salida de la RNA.        |
------------------------+-------------------------------------
|        retorno        |   Flotante del valor del error     |
|                       |   cuadratico medio de la RNA. Es   |
|                       |   0.5 por la sumatoria del cuadrado|
|                       |   de errores individuales.         |
--------------------------------------------------------------
"""
def error_cuadratico_medio(e):
    return (1./2.)*(sum(e**2.))

"""
Define el error individual de cada neurona de la red

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|           y           |   Flotante del valor esperado de   |
|                       |   salida de la neurona evaluada.   |
------------------------+-------------------------------------
|           h           |   Flotante del valor de salida de  |
|                       |   de la neurona evaluada.          |
------------------------+-------------------------------------
|        retorno        |   Flotante que indica la           |
|                       |   diferencia de valores, por lo    |
|                       |   tanto, el error individual.      |
--------------------------------------------------------------
"""
def error_individual(y,h):
    return y-h

"""
Energia de la neurona.

--------------------------------------------------------------
|       Parametro       |           Descripción              |
------------------------+-------------------------------------
|           x           |   Vector de entradas flotantes de  |
|                       |   la neurona evaluada.             |
------------------------+-------------------------------------
|           w           |   Vector de pesos flotantes de la  |
|                       |   neurona evaluada.                |
------------------------+-------------------------------------
|        retorno        |   Flotante resultante del producto |
|                       |   punto de los vectores x y w.     |
--------------------------------------------------------------
"""
def energia(x, w):
    return x.dot(w)

#data: contiene datos del archivo datos.csv
data = sp.genfromtxt("./datos.csv", delimiter = " ")

#t: Vector de respuestas esperadas, se genera
#   a partir de la ultima columna de data.
t = data[:,[-1]]

#p: Vector de datos, se genera a partir de las
#   dos primeras columnas de data.
p = data[:,:-1]

#b: Vector de unos de ganancia. Se genera del
#   tamaño de p
b = np.ones((p.shape[0], 1))
#Se concatena b a la matriz p
p = np.concatenate((b,p), axis = 1)

#Se generan pesos aleatorios para iniciar
#o se inician con algunos estipulados.
# w1 = np.random.rand(p.shape[1], 1)
# w2 = np.random.rand(p.shape[1], 1)
# w3 = np.random.rand(p.shape[1], 1)
w1 = np.array([[0.5], [-0.5], [0.8]])
w2 = np.array([[0.9], [0.4], [-0.2]])
w3 = np.array([[0.5], [0.8], [-0.1]])

print "P: {}".format(p)
print "t: {}".format(t)
print "Pesos iniciales:"
print "w1: {}\nw2: {}\nw3: {}".format(w1,w2,w3)

#j: numero de iteraciones.
j = 0

#error_total: arreglo de promedio de errores por epoca,
#             sirve para graficar al final.
error_total = []

#Este ciclo itera al terminar toda la matriz de datos
#La iteracion se hace por toda la tabla.
while True:
    #errores: contador de errores de la epoca.
    errores = 0

    #i: iterador de la epoca.
    i = 0

    #promedio: guarda la suma de los errores de
    #          la epoca.
    promedio = 0

    #Se obtiene una fila "x" de "p" por iteracion
    #Este ciclo se itera por cada fila de "p"
    for x in p:

        print "Iteracion {}".format(j)
        j+=1

        #Se imprimen los valores a utilizar en esta iteracion
        print "X: {}".format(x)

        #Se guarda la funcion de transferencia de la neurona 1, K-1.
        energia_n1 = energia(x,w1)
        #n1: salida de la neurona 1, K-1.
        n1 = transferencia(energia_n1)
        print "Entrada 1 N3: {}".format(n1)

        #Se guarda la funcion de transferencia de la neurona 2, K-1.
        energia_n2 = energia(x,w2)
        #n2: salida de la neurona 2, K-1.
        n2 = transferencia(energia_n2)
        print "Entrada 2 N3: {}".format(n2)

        #Se generan las entradas para la tercera neurona.
        entradas_n3 = np.array([1.,n1[0],n2[0]])
        print "K-1: {}".format(entradas_n3)

        #Se guarda la funcion de transferencia de la neurona 3, K.
        energia_n3 = energia(entradas_n3,w3)
        #n3: salida de la neurona 3, K.
        n3 = transferencia(energia_n3)
        print "Respuesta: {}".format(n3)

        #Se obtiene el error individual de la salida de la Red.
        error_n3 = error_individual(t[i],n3)
        print "Error individual: {}".format(error_n3)

        #Se obtiene el Error Cuadratico Medio de la Red.
        error = error_cuadratico_medio(error_n3)
        print "Error Cuadratico Medio: {}".format(error)

        #Se suma para promedio.
        promedio += error

        #Se compara el Error Cuadratico Medio, si es mayor al error
        #maximo permitido, se corrige, de lo contrario, se itera a
        #la otra fila.
        if (error > error_permitido):

            #Si existe error, se suma al contador de errores.
            errores+=1

            #Se obtiene el gradiente de K
            gradiente_k = gradiente(error_n3, transferencia_min(n3))
            #d_o: contiene el valor de la funcion Delta Omega de la neurona 3.
            d_o = delta_omega_min(gradiente = gradiente_k, p = entradas_n3)
            #gradiente_1: Contiene el gradiente de la neurona 1.
            gradiente_1 = gradiente_local(
                                gradiente_k,
                                w3[1],
                                transferencia_min(n1)
                            )

            #d_o_1: contiene el valor de la funcion Delta Omega de la neurona 1.
            d_o_1 = delta_omega_min(gradiente = gradiente_1, p = x)
            #gradiente_2: Contiene el gradiente de la neurona 2.
            gradiente_2 = gradiente_local(
                                gradiente_k,
                                w3[2],
                                transferencia_min(n2)
                            )
            #d_o_2: contiene el valor de la funcion Delta Omega de la neurona 2.
            d_o_2 = delta_omega_min(gradiente = gradiente_2, p = x)
            print "F'(x): {}".format(transferencia_min(n3))
            print "W3_1: {} W3_2: {}".format(w3[1], w3[2])
            print "F'(x)_k-1: {}, {}".format(transferencia_min(n1), transferencia_min(n2))
            print "Gradiente K: {}\nGradiente K-1: [{},{}]".format(
                        gradiente_k,
                        gradiente_1,
                        gradiente_2
                    )

            #Se asignan los nuevos pesos a las neuronas.
            w3 = delta(w3, d_o).T
            w1 = delta(w1,d_o_1).T
            w2 = delta(w2, d_o_2).T
            print "W1: {}\nW2: {}\nW3: {}".format(w1,w2,w3)

        i+=1

    #Se agrega el promedio de errores de la epoca al arreglo.
    error_total.append(promedio/4)

    #Si el contador de errores esta a 0, se termina la iteracion.
    if not errores:
        break

#Se imprimen los pesos correctos de la Red Neuronal.
print "Los pesos correctos son:\nW3: {}\nW1: {}\nW2: {}".format(w3,w1,w2)

#Se grafica el promedio de errores de la epoca.
plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.plot(np.arange(0., len(error_total), 1.),np.array(error_total))
plt.grid(True)
plt.show()

#Se grafican los pesos de las neuronas.
I = np.arange(-5, 5, 1)
J1 = (-w1[1]*I - w1[0]) / w1[2]
J2 = (-w2[1]*I - w2[0]) / w2[2]

plt.axhline(0, color="black")
plt.axvline(0, color="black")
plt.plot([1, 0], [1, 0], "o")
plt.plot([1, 0], [0, 1], "o")
plt.plot(I,J1)
plt.plot(I,J2)
plt.grid(True)
plt.show()
