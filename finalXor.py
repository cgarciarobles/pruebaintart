import scipy as sp
import numpy as np
import timeit as ti
import matplotlib.pyplot as pl

data = sp.genfromtxt("data.csv", delimiter="")
t = data[:,[-1]]    #obtengo la ultima columna del arreglo data
p = data[:,:-1]     #obtengo el arreglo data sin ultima columna
b = np.ones((p.shape[0],1)) #genero vector de 1's
p = np.concatenate((b,p), axis = 1) #concateno el vector de 1's al inicio de p, ahora tengo 111,110,101,100
w = np.array([[0.5,-0.5,0.8],[0.9,0.4,-0.2],[0.5,0.8,-0.1]]) #arreglo de pesos; para neurona1, vector 1; para neurona 2 v2; para neurona n;vn
                                                             #[ganancia, entrada1, entrada2]
constante = 0.3

def neurona(p,w):
    #La funcion neurona realiza la multiplicacion entre el vector de entradas (o puntos, en el primer caso) y los pesos de esta neurona, representados en w[n]
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
    return 0.5*pow(sumatoria,2)

def gradiente(sumatoria, hdx):
    fprima = funcionPrima(hdx)
    return -1*sumatoria*fprima

def errorMax(errorRecibido):
    if (errorRecibido > 0.01):
        return False
    else:
        return True

def deltaOmega(gradiente,vector):
    return -constante*gradiente*vector

def correccionCO(gradiente, prima):
    return -constante*gradiente*prima

contador = 0 #Este me permite llevar la cuenta de la cantidad de iteraciones que no he corregido, al cumplirse 4, es decir una etapa, se termina y la red esta entrenada
epocas = 0
while (contador<4):
    sumatoria = 0
    epocas += 1
    print "EPOCA    :   ",epocas
    for i in range(0,4):
        hdx1 = neurona(p[i],w[0])
        hdx2 = neurona(p[i],w[1])
        vectorIng = np.array([1,hdx1,hdx2]) #El vector de ingresos de la neurona 3 es la ganancia y la salida de las neuronas 1 y 2.
        hdx3 = neurona(vectorIng,w[2])
        errorIndividual = (t[i] - hdx3) #El error individual esta determinado por la salida de la porcion de red, es el esperado menos el obtenido
        sumatoria += errorIndividual #La sumatoria se da cuando la red tiene varias salidas. En este caso, por ser una, es igual al error individual
        eCuadratico = errorCuadratico(sumatoria) #El error cuadratico se calcula a partir de la sumatoria de los errores individuales de la etapa
        gradienteN3 = gradiente(sumatoria,hdx3)
        print "Datos de entrada",p[i]
        print "Salida Neurona 1 ",hdx1
        print "Salida Neurona 2 ",hdx2
        print "Salida Neurona 3 ",hdx3
        print "Error Individual ",errorIndividual
        print "Error Cuadratico ",eCuadratico
        print "Gradiente N3     ",gradienteN3
        if (errorMax(errorCuadratico) == False):
            gradienteN1 = w[2][1] * gradienteN3 * funcionPrima(hdx1) #el gradiente de la neuronaN capa k, es el peso que la conecta con la neurona en la capa k+1 * gradiente en capa k+1 * transferenciaPrima en capa k
            gradienteN2 = w[2][2] * gradienteN3 * funcionPrima(hdx2)
            print "\tGradiente N1 ",gradienteN1
            print "\tGradiente N2 ",gradienteN2
            print "\n\tPesos en la capa K:"
            print "\t",w[2]
            w[2] = w[2] + deltaOmega(gradienteN3,vectorIng) #Los nuevos pesos de w2 son los pesos anteriores menos deltaOmega
            print "\tPesos en la capa K despues de corregir:"
            print "\t",w[2]

            fPrimaN = funcionPrima(hdx1) #Para corregir los pesos de las capas ocultas, necesitamos utilizar la funcion prima
            print "\n\tPesos en la capa K-1, Neurona 1:"
            print "\t",w[0]
            #w[0] = w[0] + correccionCO(gradienteN1,fPrimaN) #Para la correcion de una neurona necesitamos su funcion prima y el gradiente local
            w[0] = w[0] + deltaOmega(gradienteN1,w[0]) #Para la correcion de una neurona necesitamos su funcion prima y el gradiente local
            print "\tPesos en la capa K-1, Neurona 1, despues de corregir:"
            print "\t",w[0]

            fPrimaN = funcionPrima(hdx2)
            print "\n\tPesos en la capa K-1, Neurona 2"
            print "\t",w[1]
            #w[1] = w[1] + correccionCO(gradienteN2,fPrimaN)
            w[1] = w[1] + deltaOmega(gradienteN2,w[1])
            print "\tPesos en la capa K-1, Neurona 2, despues de corregir:"
            print "\t",w[1]
            print "\n\n"
            contador = 0
        else:
            contador += 1
            #print contador

        if (contador == 4):
            break;
        if (epocas == 5000):
            break;
    if (epocas == 5000):
        break;
    if (contador == 4):
        break;


print epocas
