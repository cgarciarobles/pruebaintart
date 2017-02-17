import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("datos.csv", delimiter="")
t = data[:,[-1]] #obtengo la ultima columna del arreglo data
p = data[:,:-1] #obtengo el arreglo data sin ultima columna
b = np.ones((p.shape[0],1))

p = np.concatenate((b,p), axis = 1)
#w = np.random.rand(p.shape[1],1) #genera los numeros aleatorios que representan los pesos de la neurona
w = np.array([[-0.5],[-1],[1]])

#print p
#print w
#print np.dot(p,w)

contador = 0
iterador = 0
aux = 0

def neurona (p,w):
    suma = p.dot(w)
    return int(suma[0] >= 0)

def corregir(want,er,pex):
    w = want + er*np.transpose(pex)
    return w


while (contador < 4):
    aux += 1
    error = t[iterador] - neurona(p[iterador],w)
    print 'error {}'.format(iterador) + str(error)
    if error[0]:
        contador = 0
        w = corregir(w, error, p[iterador][np.newaxis])
    else:
        print 'pit {}'.format(p[iterador])
        contador += 1

    if (iterador < 3):
        iterador = iterador + 1
    else:
        iterador = 0

print aux
x = np.arange(-5. , 5.0, 1)
y = -w[1]*x + w[0] / w[2]
pl.plot(x,y)
pl.grid(True)
pl.show()
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
