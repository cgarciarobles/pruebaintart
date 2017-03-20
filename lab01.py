def llamada(x):
    sum = 0
    for i in range(len(x)-1):
        #print x[i][0]
        sum = sum + x[i][0]*x[i+1][1]

    for i in range(1,len(x)):
        #print x[i][0]
        sum = sum + x[i][0]*x[i-1][1]

    return 0.5*sum

x = ([[2,3],[-3,1],[-3,2],[4,5]])
print llamada(x)
