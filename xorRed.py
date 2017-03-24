import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("datos.csv", delimiter="")
t = data[:,[-1]]    #obtengo la ultima columna del arreglo data
p = data[:,:-1]     #obtengo el arreglo data sin ultima columna
w1 = np.array([-0.6,-0.2])
summation   = 0.

def neurona(p,w):
        r = p.dot(w)
        y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
        return y

def gradiente(hdx):

    fPrima = np.exp(-hdx) / pow(1+np.exp(-hdx),2)
    return -1 *hdx*-fPrima

contador = 0;
iterador = 0;

while (contador < 4):
    hdx = neurona(p[iterador],w1)
    individualE = (t[iterador] - hdx)
    summation += individualE
    errorCuadratico = (0.5)*pow(summation,2)
    gradient = gradiente(hdx)
    print "error Individual:    ",individualE
    print "hdx:                 ",hdx
    print "error Cuadratico:    ",errorCuadratico
    print "gradiente:           ",gradient
    print "\n"
