#Este es un algoritmo para la correcion del error en los pesos de un una neurona

import scipy as sp
import numpy as np
import timeit
import matplotlib.pyplot as pl

# ------------


data = sp.genfromtxt("datos.csv", delimiter=" ")

t = data[:,[-1]]
p = data[:,:-1]
b = np.ones((p.shape[0],1))


p = np.concatenate( (b,p),axis=1 )
#w = np.random.rand(p.shape[1],1) #genera los numeros aleatorios que representan los pesos de la neurona
#w = np.array([[-0.5],[-1],[1]])
#w = np.array([[0.80922424],[0.42854118],[0.34443655]]) #pruebas para w
w = np.array([[0.1962092],[0.22406114],[0.55542444]])

def neurona (p,w):
    suma = p.dot(w)
    return int(suma[0] >= 0)

def corregir(want,error,p):
    w = want+error*np.transpose(p)
    return w

#error = t[2]-neurona(p[2],w)

# if error:
#     w = corregir(w,error,p[2])
j=0
while True:
    i=0
    errores = 0
    for point in p:
        print 'Iteracion',j
        binario = neurona(point, w)
        error1 = t[i] - binario
        print w
        if error1 != 0:
            errores+=1
            w = corregir(w,error1,p[i][np.newaxis])
            print w

        print 'esperado',t[i]
        print 'error',error1
        print 'binario',binario
        i=i+1
        j+=1
    if not errores:
        break

print w

x = np.arange(-5. , 5.0, 1)
y = -w[1]*x - w[0] / w[2]

pl.plot(1,1,"o")
pl.plot(0,1,"o")
pl.plot(1,0,"o")
pl.plot(0,0,"o")
pl.plot(x,y)
pl.grid(True)
pl.show()
#print t
# print p[2]
# print b
# print w
# # print neurona(p[2],w)
# print p.shape[0]
# print p
