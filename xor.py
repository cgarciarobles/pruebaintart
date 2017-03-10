import numpy as np
import matplotlib.pyplot as pl


#x = np.arange(-10,10,0.1)
#y = 1 / (1 + np.exp(-x));j = (np.exp(-x)) / pow((1 + np.exp(-x)),2)
#pl.plot(x,y)
#pl.plot(x,j)
#pl.show()

x = np.array([0.,0.]) #vector de ingreso para neurona
w = np.array([0.3,-0.1]) #pesos que se multiplican por el vector que ingresa

def neurona(x,w):
    r = x.dot(w)
    y = 1 / (1 + np.exp(-r));j = (np.exp(-r)) / pow((1 + np.exp(-r)),2)
    print y

neurona(x,w)
