import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("data.csv", delimiter="")
t = data[:,[-1]]    #obtengo la ultima columna del arreglo data
p = data[:,:-1]     #obtengo el arreglo data sin ultima columna

print p
b = np.ones((p.shape[0],1))
p = np.concatenate((b,p), axis = 1)
print p


w1 = np.array([[0.5,-0.5,0.8],[0.9,0.4,-0.2],[0.5,0.8,-0.1]])
summation   = 0.

def neurona(p,w):
        r = p.dot(w)
        y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
        return y

def gradiente(sumatoriaEI, hdx):
    #fPrima = np.exp(-hdx) / pow(1+np.exp(-hdx),2)
    salida = transferencia(hdx)
    return -1*sumatoriaEI*salida

def errorMax(errorRecibido):

    #diferencia = 0.1 - errorRecibido
    if ((errorRecibido >= -0.10) and (errorRecibido <= 0.10)) or ((errorRecibido >= 0.90) and (errorRecibido <= 1.10)) :
        return True
    else:
        return False


def proceso(wa, contador, iterador):
    summation = 0.
    print "hola"
    print p[iterador]
    hdx1 = neurona(p[iterador],wa[0])
    ##individualE1 = (t[iterador] - hdx1)  #para borrar
    ##summation += individualE1  #para borrar
    ##gradient1 = gradiente(individualE1,hdx1)  #para borrar
    ##errorCuadratico = (0.5)*pow(summation,2)  #para borrar
    #print "error Individual:    ",individualE1  #para borrar

    print "hdx1:                 ",hdx1
    ##print "gradiente:           ",gradient1  #para borrar

    print "\n"

    #
    hdx2 = neurona(p[iterador],wa[1])
    ##individualE2 = (t[iterador] - hdx2)  #para borrar
    ##summation += individualE2  #para borrar
    #gradient2 = gradiente(individualE2,hdx2)  #para borrar
    #errorCuadratico = (0.5)*pow(summation,2)  #para borrar
    #print "error Individual:    ",individualE2   #para borrar
    print "hdx2:                 ",hdx2
    #print "gradiente:           ",gradient2  #para borrar

    print "\n"

    #
    vector = np.array([1,hdx1,hdx2])
    hdx3 = neurona(vector,wa[2])
    individualE3 = (t[iterador] - hdx3)
    summation += individualE3
    gradient3 = gradiente(individualE3, hdx3)
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
        gradient1 = wa[2][1] * gradient3 * transferencia(hdx1)
        gradient2 = wa[2][2] * gradient3 * transferencia(hdx2)
        #print gradient1
        #print wa[2]
        wa[2] = wa[2] - deltaOmega(wa,gradient3,vector)
        print wa[2]
        #aux = ftransferencia(p[iterador])
        aux = transferencia(hdx1)
        print wa[0]
        wa[0] = wa[0] - corregirO(gradient1, aux)
        print wa[0]

        aux = transferencia(hdx2)
        wa[1] = wa[1] - corregirO(gradient2, aux)
        #wa[0] = corregirO(gradient3, wa[0], aux)
        #wa[1] = corregirO(gradient3, wa[1], aux)
        #if (iterador == 0):
        w1[0] = wa[0]
        w1[1] = wa[1]
        w1[2] = wa[2]



def deltaOmega(wa, gradient, vec):
    return 0.5*gradient*vec

def corregirO(gradient, aux):
    return 0.5*gradient*aux

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
    print "ITERACION",caux
    if (caux == 4000):
        break;
    if (iterador < 3):
        #if (iterador == 0):
        if (proceso(w1, contador, iterador) == 0):
            contador += 1
        else:
            contador = 0
        iterador = iterador + 1
    else:
        if (proceso(w1, contador, iterador) == 0):
            contador += 1
        else:
            contador = 0
        iterador = 0
