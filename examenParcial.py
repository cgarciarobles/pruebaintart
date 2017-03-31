import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("archivoEntrada.csv", delimiter="")
p = data[:,:-2]             #obtengo el arreglo sin las ultimas dos columnas
t = data[:,[-2,-1]]         #obtengo las ultimas dos columnas
constante = 0.3
erroresSum = []
w = np.array([[0.5,-0.5,0.8, -0.5],[0.9,0.4,-0.2, -0.2],[0.5,0.2,0.1,0.8], #Capa1 N1, N2, N3
                [0.5,-0.5,0.8, -0.5],[0.9,0.4,-0.2,-0.2],[0.5,0.2,0.1,0.8],    #Capa2 N4, N5, N6
                [0.5,0.8,-0.1,0.7],[0.5,0.7,-0.2,0.4]])                        #Capa3 N7, N8

#Ganancia
b = np.ones((p.shape[0],1)) #genero vector de 1's
p = np.concatenate((b,p), axis = 1) #concateno el vector de 1's al inicio de p

def neurona(p,w):
    #La funcion neurona realiza la multiplicacion entre el vector de entradas (ENERGIA)
    #(o puntos, en el primer caso) y los pesos de esta neurona, representados en w[n].
    #Es decir, la transferencia de la energia
    energy = energia(p,w)
    y = transferencia(energy)
    return y #y = hdx

def energia(p,w):
    #La energia de una neurona es la sumatoria de la multiplicacion de W/k+1/ * h/k/
    energy = p.dot(w)
    return energy

def transferencia(energy):
    #La transferencia en la capa k+1, funciona respecto la energia de la neurona k+1
    y = 1 / (1 + np.exp(-energy));j = (np.exp(-energy)) / pow((1 + np.exp(-energy)),2)
    return y

def funcionPrima(hdx):
    #La funcion de transferencia esta en terminos de la salida (y o hdx)
    prima = hdx - pow(hdx,2)
    return prima

def funcionPrimaLarga(x):
    prima = np.exp(-x) / pow((1+np.exp(-x)),2)
    print prima
    prima = (np.exp(-x)) / (1+np.exp(-x))**2
    print prima
    return prima

def errorCuadratico(sumatoria):
    return 0.5*(sumatoria)

def gradiente(sumatoria, hdx):
    fprima = funcionPrima(hdx)
    return -1*sumatoria*fprima

def cambioPeso(peso,gradiente,entrada):
    return peso.T + (-0.3)*gradiente*entrada.T

def deltaOmega(gradiente,vector):
    return -constante*gradiente*vector

epocas = 0
while True:
    contador = 0            #Este me permite llevar la cuenta de la cantidad
                            #de iteraciones que no he corregido, al cumplirse 4,
                            #es decir una etapa, se termina y la red esta entrenada
    epocas += 1
    promedio = 0
    print "EPOCA    :   ",epocas
    for i in range(0,4):
        sumatoria = 0
        hdx1 = neurona(p[i],w[0])
        hdx2 = neurona(p[i],w[1])
        hdx3 = neurona(p[i],w[2])
        entradasN1 = np.array([1,hdx1,hdx2,hdx3])
        hdx4 = neurona(entradasN1, w[3])
        hdx5 = neurona(entradasN1, w[4])
        hdx6 = neurona(entradasN1, w[5])
        entradasN2 = np.array([1,hdx4,hdx5,hdx6])  #El vector de ingresos de las neurona 7 y 8
                                                    #es la ganancia y la salida de las neuronas 4,5,6
        hdx7 = neurona(entradasN2, w[6])
        hdx8 = neurona(entradasN2, w[7])
        errorIndividualN7 = (t[i][0] - hdx7)
        errorIndividualN8 = (t[i][1] - hdx8)
        sumatoria += pow(errorIndividualN7,2) + pow(errorIndividualN8,2)
        eCuadratico = errorCuadratico(sumatoria)
        sumatoria = errorIndividualN7 + errorIndividualN8
        gradienteN7 = gradiente(sumatoria, hdx7)
        gradienteN8 = gradiente(sumatoria, hdx8)
        promedio += eCuadratico

        #SECCION DE IMPRESION
        print "Datos de entrada",p[i]
        print "Salida Neurona 1 ",hdx1
        print "Salida Neurona 2 ",hdx2
        print "Salida Neurona 3 ",hdx3
        print "Salida Neurona 4 ",hdx4
        print "Salida Neurona 5 ",hdx5
        print "Salida Neurona 6 ",hdx6
        print "Salida Neurona 7 ",hdx7
        print "Salida Neurona 8 ",hdx8
        print "Error Individual N7 ",errorIndividualN7
        print "Error Individual N8 ",errorIndividualN8
        print "Error Cuadratico ",eCuadratico
        print "Gradiente N7     ",gradienteN7
        print "Gradiente N8     ",gradienteN8
        #FIN SECCION DE IMPRESION

        if eCuadratico > 0.01:
            gradienteN4 = (w[6][1] * gradienteN7 * funcionPrima(hdx4)  + w[7][1] * gradienteN8 * funcionPrima(hdx4))
                #el gradiente de la neuronaN capa k, es el peso que la conecta con la neurona en la capa k+1 *
                #gradiente en capa k+1 * transferenciaPrima en capa k.
                #En este caso es w[peso de la capa k+1][conexion con neurona de la capa k+1] * gradiente capa k+1 * fprima de la salida de la neurona
            gradienteN5 = (w[6][2] * gradienteN7 * funcionPrima(hdx5)  + w[7][2] * gradienteN8 * funcionPrima(hdx5))
            gradienteN6 = (w[6][3] * gradienteN7 * funcionPrima(hdx6)  + w[7][3] * gradienteN8 * funcionPrima(hdx6))
            gradienteN1 = (w[3][1] * gradienteN4 * funcionPrima(hdx1)  +  w[4][1] * gradienteN5 * funcionPrima(hdx1) + w[5][1] * gradienteN6 * funcionPrima(hdx1))
            gradienteN2 = (w[3][2] * gradienteN4 * funcionPrima(hdx2)  +  w[4][2] * gradienteN5 * funcionPrima(hdx2) + w[5][2] * gradienteN6 * funcionPrima(hdx2))
            gradienteN3 = (w[3][3] * gradienteN4 * funcionPrima(hdx3)  +  w[4][3] * gradienteN5 * funcionPrima(hdx3) + w[5][3] * gradienteN6 * funcionPrima(hdx3))


            #ULTIMAS DOS NEURONAS
            print "\n\tPesos en la capa K:"
            print "\t",w[6] #Pesos de las salidas, ultimas dos neuronas
            print "\t",w[7]
            w[6] = w[6] + deltaOmega(gradienteN7,entradasN2)
            w[7] = w[7] + deltaOmega(gradienteN8,entradasN2)
            print "\tPesos en la capa K despues de corregir:"
            print "\t",w[6]
            print "\t",w[7]
            print '\n'


            print "Gradiente N4: ",gradienteN4
            print "Gradiente N5: ",gradienteN5
            print "Gradiente N6: ",gradienteN6
            #TRES NEURONAS INTERMEDIAS
            print entradasN1
            print "\n\tPesos en la capa K-1, Neurona 1:"
            print "\t",w[3]
            w[3] = cambioPeso(w[3],gradienteN4,entradasN1).T
            print "\tPesos en la capa K-1, Neurona 1, despues de corregir:"
            print "\t",w[3]

            print "\n\tPesos en la capa K-1, Neurona 2:"
            print "\t",w[4]
            w[4] = cambioPeso(w[4],gradienteN5,entradasN1).T
            print "\tPesos en la capa K-1, Neurona 2, despues de corregir:"
            print "\t",w[4]

            print "\n\tPesos en la capa K-1, Neurona 3:"
            print "\t",w[5]
            w[5] = cambioPeso(w[5],gradienteN6,entradasN1).T
            print "\tPesos en la capa K-1, Neurona 3, despues de corregir:"
            print "\t",w[5]
            print "\n"

            print "Gradiente N1: ",gradienteN1
            print "Gradiente N2: ",gradienteN2
            print "Gradiente N3: ",gradienteN3
            #PRIMERAS TRES NEURONAS
            print "\n\tPesos en la capa K-2, Neurona 1:"
            print "\t",w[0]
            w[0] = cambioPeso(w[0],gradienteN1,p[i]).T
            print "\tPesos en la capa K-2, Neurona 1, despues de corregir:"
            print "\t",w[0]

            print "\n\tPesos en la capa K-2, Neurona 2:"
            print "\t",w[1]
            w[1] = cambioPeso(w[1],gradienteN2,p[i]).T
            print "\tPesos en la capa K-2, Neurona 2, despues de corregir:"
            print "\t",w[1]

            print "\n\tPesos en la capa K-2, Neurona 3:"
            print "\t",w[2]
            w[2] = cambioPeso(w[2],gradienteN3,p[i]).T
            print "\tPesos en la capa K-2, Neurona 3, despues de corregir:"
            print "\t",w[2]
            print '\n'

            contador = 0
        else:
            contador += 1
            print "\n\n"

        if contador == 4:
            break;

    erroresSum.append(promedio/4)
    if (epocas == 1000):
        break;
    if (contador == 4):
        break;

print epocas
pl.axhline(0, color="black")
pl.axvline(0, color="black")
pl.plot(np.arange(0., len(erroresSum), 1.),np.array(erroresSum))
pl.grid(True)
pl.show()
