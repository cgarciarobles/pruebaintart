import numpy as np
import matplotlib.pyplot as pl


#x = np.arange(-10,10,0.1)
#y = 1 / (1 + np.exp(-x));j = (np.exp(-x)) / pow((1 + np.exp(-x)),2)
#pl.plot(x,y)
#pl.plot(x,j)s
#pl.show()

#x = np.array([0.,0.]) #vector de ingreso para neurona
#w = np.array([[0.3,-0.1], [0.5,0.8]]) #pesos que se multiplican por el vector que ingresa
x = np.array([0.5,0.5])
#w = np.array([[-0.6,-0.2]])
w = np.array([[-0.8,0.5],[0.2,0.1],[0.4,-0.9]])


def neurona(x,w):
    for i in range(len(w)):
        r = x.dot(w[i])
        print x
        print w
        print r
        y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
        return y


def errorcuadratico(errCuad):
    print "error cuadratico medio (1/2)(SUM errores)^2:         ",(0.5)*pow(errCuad,2)


# def gradiente(x,w):
#     return -1*0.401312339888*-fp(transferencia(x,w),x)
#
#
# def transferencia(x,w):
#     return x*np.transpose(w)
#
#
# def fp(energia, x):
#     return (np.exp(-x))/ pow((1+np.exp(-x)),2)

errIndv = 0 - neurona(x,w)
errCuad =  neurona(x,w) #y(x) - h(x)
errorcuadratico(errCuad)

hdx = neurona(x,w)

print "h(x) o valor de la neurona:                      ",hdx #o neurona(x,w)
print "error individual de la neurona y(x)-h(x):            ",errIndv
#print gradiente(x,w)

print "\ncalculo para gradiente..............\n"
