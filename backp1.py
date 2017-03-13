import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("data.csv", delimiter="")
t = data[:,[-1]]    #obtengo la ultima columna del arreglo data
p = data[:,:-1]     #obtengo el arreglo data sin ultima columna
w = np.array([[-0.6,-0.2], [-0.6,-0.2]])
summation   = 0.

def neurona(p,w):
    #print "w tam: ",len(w)
    #print "range: ",range(len(w))
    #for i in range(len(w)):
        #print "i: ",i
        r = p.dot(w)
        #print "p:",p
        #print "w:",w
        #print "r",r
        y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
        return y

def gradiente(hdx):
    #print "w:",w
    #print "wtrans:",np.transpose(w)[np.newaxis]
    fPrima = np.exp(-hdx) / pow(1+np.exp(-hdx),2)
    return -1 *hdx*-fPrima

for i in range(len(w)):
    hdx = neurona(p[i],w[i])
    individualE = (t[i] - hdx)
    summation += individualE
    errorCuadratico = (0.5)*pow(summation,2)
    gradient = gradiente(hdx)
    print "error Individual:    ",individualE
    print "hdx:                 ",hdx
    print "error Cuadratico:    ",errorCuadratico
    print "gradiente:           ",gradient
    print "\n"
