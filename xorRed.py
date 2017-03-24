import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("data.csv", delimiter="")
t = data[:,[-1]]    #obtengo la ultima columna del arreglo data
p = data[:,:-1]     #obtengo el arreglo data sin ultima columna

b = np.ones((p.shape[0],1))
p = np.concatenate((b,p), axis = 1)

#w1 = np.array([-0.6,-0.2])
#w2 = np.array([-0.6,-0.2])
#w3 = np.array([-0.6,-0.2])
w1 = np.array([[-0.8,0.5],[0.2,0.1],[0.4,-0.9]])
#w2 = np.array([[-0.8,-0.2],[-0.6,-0.2],[-0.6,-0.2]])
w2 = np.array([[-0.8,0.5],[0.2,0.1],[0.3730,-0.92]])
w3 = np.array([[-0.8,0.5],[0.2,0.1],[0.3730,-0.92]])
w4 = np.array([[-0.8,0.5],[0.2,0.1],[0.3730,-0.92]])
#w3 = np.array([[-0.6,-0.2],[-0.6,-0.2],[-0.6,-0.2]])
#w4 = np.array([[-0.6,-0.2],[-0.6,-0.2],[-0.6,-0.2]])



#pruebas clase
w1 = np.array([[-0.5,0.7],[0.2,-0.3],[-1.3,-3]])
w2 = np.array([[-0.5,0.7],[0.2,-0.3],[-1.3,-3]])
w3 = np.array([[-0.5,0.7],[0.2,-0.3],[-1.3,-3]])
w4 = np.array([[-0.5,0.7],[0.2,-0.3],[-1.3,-3]])

summation   = 0.

def neurona(p,w):
        r = p.dot(w)
        y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
        return y

def gradiente(hdx):

    fPrima = np.exp(-hdx) / pow(1+np.exp(-hdx),2)
    return -1 *hdx*-fPrima

def errorMax(errorRecibido):
    diferencia = 0.1 - errorRecibido
    if (diferencia >= 0):
        return True
    else:
        return False


def proceso(wa, contador, iterador):
    summation = 0.
    print p[iterador]
    hdx1 = neurona(p[iterador],wa[0])
    individualE1 = (t[iterador] - hdx1)
    summation += individualE1
    gradient1 = gradiente(hdx1)
    errorCuadratico = (0.5)*pow(summation,2)
    print "error Individual:    ",individualE1
    print "hdx1:                 ",hdx1
    print "gradiente:           ",gradient1
    #print "error Cuadratico:    ",errorCuadratico
    print "\n"
    #
    hdx2 = neurona(p[iterador],wa[1])
    individualE2 = (t[iterador] - hdx2)
    summation += individualE2
    gradient2 = gradiente(hdx2)
    errorCuadratico = (0.5)*pow(summation,2)
    print "error Individual:    ",individualE2
    print "hdx2:                 ",hdx2
    print "gradiente:           ",gradient2
    #print "error Cuadratico:    ",errorCuadratico
    print "\n"
    #
    vector = np.array([hdx1,hdx2])
    hdx3 = neurona(vector,wa[2])
    individualE3 = (t[iterador] - hdx3)
    summation += individualE3
    gradient3 = gradiente(hdx3)
    errorCuadratico = (0.5)*pow(summation,2)
    print "error Individual:    ",individualE3
    print "hdx3:                 ",hdx3
    print "gradiente:           ",gradient3
    print p[iterador]
    print "error Cuadratico:    ",errorCuadratico
    print "\n"
    #
    if errorMax(errorCuadratico):
        return 1
    else:
        #vec = np.array([[hdx1,hdx2]])
        vec = p[iterador]
        wa[2] = wa[2] - deltaOmega(wa,gradient3,vec)
        #aux = ftransferencia(p[iterador])
        aux = transferencia(hdx1)
        wa[0] = wa[0] - corregirO(gradient1, p[iterador], aux)
        aux = transferencia(hdx2)
        wa[1] = wa[1] - corregirO(gradient1, p[iterador], aux)
        #wa[0] = corregirO(gradient3, wa[0], aux)
        #wa[1] = corregirO(gradient3, wa[1], aux)
        if (iterador == 0):
            w1[0] = wa[0]
            w1[1] = wa[1]
            w1[2] = wa[2]
        elif (iterador == 1):
            w2[0] = wa[0]
            w2[1] = wa[1]
            w2[2] = wa[2]
        elif (iterador == 2):
            w3[0] = wa[0]
            w3[1] = wa[1]
            w3[2] = wa[2]
        elif (iterador == 3):
            w4[0] = wa[0]
            w4[1] = wa[1]
            w4[2] = wa[2]


def deltaOmega(wa, gradient, vec):
    return 0.5*gradient*vec

def corregirO(gradient, wa, aux):
    return gradient*wa*aux

def ftransferencia(x):
    print "traaaaansferencia",np.exp(-x) / pow(1+np.exp(-x),2)
    return np.exp(-x) / pow(1+np.exp(-x),2)

def transferencia(salida):
    a = salida - pow(salida,2)
    return a

contador = 0
iterador = 0
caux = 0
while (contador < 4):
    #En este marco debo trabajar las iteraciones a la tabla de entradas
    caux += 1
    if (caux == 1000):
        break;
    if (iterador < 3):
        if (iterador == 0):
            if (proceso(w1, contador, iterador) == 0):
                contador += 1
            else:
                contador = 0
        elif (iterador == 1):
            if (proceso(w2, contador, iterador) == 0):
                contador += 1
            else:
                contador = 0
        elif (iterador == 2):
            if (proceso(w3, contador, iterador) == 0):
                contador += 1
            else:
                contador = 0
        iterador = iterador + 1

    else:
        if (proceso(w4, contador, iterador) == 0):
            contador += 1
        else:
            contador = 0
        iterador = 0
