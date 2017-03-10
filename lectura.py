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
w = np.array([[1],[1],[1]])
#w = np.array([[0.80922424],[0.42854118],[0.34443655]]) #pruebas para w
#w = np.array([[0.1962092],[0.22406114],[0.55542444]])

vw = w
listax = []
listay = []
#print p
#print w
#print np.dot(p,w)

contador = 0
iterador = 0
aux = 0
sumatoria = 0

def neurona (p,w):
    suma = p.dot(w)
    return suma[0]

def corregir(want,er,pex):
    w = want + er*np.transpose(pex)
    return w

def costo(x,y): #vwa es el arreglo vw que modificare

    #print t[0]
    ran = np.arange(-5,5,0.3)
    j = (1./2.)*len(x)
    cos = 0
    for i in range(len(x)):
        cos += ((ran*x[i] - y[i])**2)

    j = j*cos
    pl.plot(ran,j)
    pl.grid(True)
    pl.show()
    #print vw[omega]
    #j = (pex[0,omega] * vw[omega])
    #print "J:",j
    #print "E:",er
    #sumatoria += er**2
    #listax.append(vw[omega])
    #listay.append(sumatoria)

# while (contador < 4):
#     x = np.arange(-5. , 5.0, 1)
#     y = (w[1]*x + w[0]) / w[2]
#     #pl.plot(x,y)
#     #pl.grid(True)
#     #pl.show(block=False)
#     aux += 1
#     error = t[iterador] - neurona(p[iterador],w)
#     print 'error {}'.format(iterador) + str(error)
#     if error[0]:
#         contador = 0
#         w = corregir(w, error, p[iterador][np.newaxis])
#         contador += 1
#     else:
#         print 'pit {}'.format(p[iterador])
#         contador += 1
#
#     if (iterador < 3):
#         iterador = iterador + 1
#     else:
#         iterador = 0
#
#
#  x = np.arange(-5 , 5.0, 1)
#  y = (-w[1]*x - w[0]) / w[2]
#
# pl.plot(1,1,"o")
# pl.plot(0,1,"o")
# pl.plot(1,0,"o")
# pl.plot(0,0,"o")
# # pl.plot(x,y)
# pl.grid(True)
# pl.show()
#
# print "\n" + "vector inicial: " + "\n{}".format(vw)
# print "Cantidad de iteraciones para el entrenamiento: {}".format(aux) + "\n"
# omega = input('Respecto a que Omega desea realizar la grafica de error (Escriba un numero [0-n]): ')

cont2 = 0
it2 = 0
aux2 = 0

while (cont2 < 4):
    aux += 1
    k = p[:,[-2]]
    costo(t,k)
    #print 'error {}'.format(it2) + str(error)
    if error[0]:
        cont2 = 0
        vw = corregir(vw, error, p[it2][np.newaxis])
        cont2 += 1
    else:
        #print 'pit {}'.format(p[it2])
        cont2 += 1

    if (it2 < 3):
        it2 = it2 + 1
    else:
        it2 = 0

print w
#fig= pl.figure()
#axes=fig.add_subplot(111)
#axes.plot(listax,listay)

#pl.show()

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

#derivacion (x/m)(wx-y)
#tratado como punto (w1x - y)x si parantesis es error, x es punto de entrada
#(w1x - y)x = 0 donde el despeje es la linea que valida y = xw
