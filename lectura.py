import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("datos.csv", delimiter="")
t = data[:,[-1]] #obtengo la ultima columna del arreglo data
p = data[:,:-1] #obtengo el arreglo data sin ultima columna
b = np.ones((p.shape[0],1))

p = np.concatenate((b,p), axis = 1)
w = np.random.rand(p.shape[1],1) #genera los numeros aleatorios que representan los pesos de la neurona

#print p
#print w
#print np.dot(p,w)

contador = 0
iterador = 0

def neurona (p,w):
    suma = p.dot(w)
    return int(suma[0] >= 0)

def corregir(er,p):
    print er
    print p
    print er*p
    print w
    print w + er*p

    return w+er*p


while (contador < 4):
    error = t[iterador] - neurona(p[iterador],w)
    print 'error {}'.format(iterador) + str(error)
    if error:
        contador = 0
        w = corregir(error, p[iterador])
    else:
        contador += 1
    
    if (iterador < 3):
        iterador = iterador + 1
    else:
        iterador = 0

# array[inicio:final:pasos]
# [0,[-1]]
# [0::-1]  primera fila
# [::-1]   filas inversas
# [-1::]   ultima fila
# [:,[-1]] ultima columna
# [:,[0]]  primera columna
# [:,:-1]  sin la ultima columna

# p.shape(0) devuelve el tamano
# np concatenate (concatena la parte b con la p, poniendo primero la b)
#       axis se refiere al eje a utiilizar 1,eje y; 0, eje x
