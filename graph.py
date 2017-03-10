import numpy as np
import matplotlib.pyplot as pl
x = np.arange(-10,10,0.1)
y = 1 / (1 + np.exp(-x));j = (np.exp(-x)) / pow((1 + np.exp(-x)),2)
pl.plot(x,y)
pl.plot(x,j)
pl.show()
