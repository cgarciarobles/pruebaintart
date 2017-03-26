import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("data.csv", delimiter="")
t = data[:,[-1]]    #obtengo la ultima columna del arreglo data
p = data[:,:-1]     #obtengo el arreglo data sin ultima columna
w = np.array([[-0.8,0.5], [0.2,-0.1],[0.4,-0.9]])
summation   = 0.

def neurona1(p,w):
    r = p.dot(w)
    y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
    return y


def neurona2(p,w):
    r = p.dot(w)
    y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
    return y

def neurona3(p,w):
    r = p.dot(w)
    y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
    return y

def gradiente(hdx):
    fPrima = np.exp(-hdx) / pow(1+np.exp(-hdx),2)
    salida = hdx - pow(hdx,2)
    #fPrima = -1 *hdx*-fPrima
    print "fPrima",fPrima
    print "fPrimaCompr",salida

    return -1 *hdx*-fPrima

def errorMax(individualE):
    diferencia = 0.05 - individualE
    if (diferencia >= 0):
        return True
    else:
        return False

def corregir(wAnt, individualE, p):
    w = wAnt + individualE*np.transpose(p)
    return w



individualE = 32000.
hdx1 = 0.
errorCuadratico = 0.
gradient = 0.


while (errorMax(individualE) == False):
    for i in range(len(w)):
        hdx1 = neurona1(p[i],w[i])
        individualE = (t[i] - hdx1)
        summation += individualE
        errorCuadratico = (0.5)*pow(summation,2)
        gradient = gradiente(hdx1)
        print "Neurona 1"
        print "error Individual:    ",individualE
        print "hdx1:                 ",hdx1
        print "error Cuadratico:    ",errorCuadratico
        print "gradiente:           ",gradient
        print "\n"
        w[i] = corregir(w[i], individualE, p[i])

individualE = 32000.
while (errorMax(individualE) == False):
    for i in range(len(w)):
        hdx2 = neurona2(p[i],w[i])
        individualE = (hdx1 - hdx2)
        summation += individualE
        errorCuadratico = (0.5)*pow(summation,2)
        gradient = gradiente(hdx2)
        print "Neurona 2"
        print "error Individual:    ",individualE
        print "hdx2:                 ",hdx2
        print "error Cuadratico:    ",errorCuadratico
        print "gradiente:           ",gradient
        print "\n"
        w[i] = corregir(w[i], individualE, p[i])

individualE = 32000.
while (errorMax(individualE) == False):
    for i in range(len(w)):
        hdx3 = neurona2(p[i],w[i])
        individualE = (hdx1 - hdx3)
        summation += individualE
        errorCuadratico = (0.5)*pow(summation,2)
        gradient = gradiente(hdx3)
        print "Neurona 3"
        print "error Individual:    ",individualE
        print "hdx3:                 ",hdx3
        print "error Cuadratico:    ",errorCuadratico
        print "gradiente:           ",gradient
        print "\n"
        w[i] = corregir(w[i], individualE, p[i])
