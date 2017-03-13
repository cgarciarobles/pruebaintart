import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("data.csv", delimiter="")
t = data[:,[-1]]    #obtengo la ultima columna del arreglo data
p = data[:,:-1]     #obtengo el arreglo data sin ultima columna
w = np.array([[-0.6,-0.2], [-0.6,-0.2]])

def neurona(p,w):
    #print "w tam: ",len(w)
    #print "range: ",range(len(w))
    for i in range(len(w)):
        #print "i: ",i
        r = p[i].dot(w[i])
        #print "p:",p[i]
        #print "w:",w[i]
        #print "r",r
        y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
        print y

hdx = neurona(p,w)
print hdx
