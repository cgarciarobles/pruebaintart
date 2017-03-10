import numpy as np
import matplotlib.pyplot as pl


#x = np.arange(-10,10,0.1)
#y = 1 / (1 + np.exp(-x));j = (np.exp(-x)) / pow((1 + np.exp(-x)),2)
#pl.plot(x,y)
#pl.plot(x,j)
#pl.show()

#x = np.array([0.,0.]) #vector de ingreso para neurona
#w = np.array([[0.3,-0.1], [0.5,0.8]]) #pesos que se multiplican por el vector que ingresa
x = np.array([0.5,0.5])
w = np.array([[-0.6,-0.2]])

def neurona(x,w):
    for i in range(len(w)):
        r = x.dot(w[i])
        y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
        print y

neurona(x,w)
