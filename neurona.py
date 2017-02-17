x = ((0.,0.), (0.,1.), (1.,0.), (1.,1.)) #vector de ingreso para neurona
w = (1,1) #pesos que se multiplican por el vector que ingresa
b = -1.5

def neurona(x,w,b):
    resultado = 0.
    for i in range(len(x)):
        resultado += x[i] * w[i]
        print x[i] * w[i]
    resultado += b
    return int(resultado >= 0)

for x1 in x:
    print "x1: {} y:{}".format(x1,neurona(x1,w,b))
