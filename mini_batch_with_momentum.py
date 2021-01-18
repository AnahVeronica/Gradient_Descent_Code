import numpy as np
from sklearn.utils import shuffle

#Data points
'''
Generates random 
x values
'''
x = np.linspace(0,5, 500)

#True Function
'''
Creates a functions of f(x) = y
'''
bo = 2
b1 = 3.14
y = bo + b1*x

def shuffled_data(x): #Shuffles the data for gradient descent
    x = shuffle(x)
    y = bo + b1*x
    return x, y

#Loss function to compute error
def loss_function(bo, b1, y, x):
    squared_error = (y - (bo + b1*x))**2
    return np.mean(squared_error)

def dervative(x, y, bo, b1): #Computes the gradient
    bo_derv = 2*(-y + bo + b1*x)
    b1_derv = 2*x*(-y + bo + b1*x)
    return bo_derv, b1_derv

def gradient_descent(X, Y, bo, b1, eta=0.0001, tolerance=0.0001, epochs = 1, beta = 0.9, batch_size = 16):
    prev_bo = 0
    prev_b1 = 0
    vdw = 0
    vdb = 0

    t = 0

    bo_values = [bo]
    b1_values = [b1]

    for i in range(epochs):
        dbo = 0
        db1 = 0
        t += 1

        X, Y = shuffle(X, Y)
        X_batch = X[:batch_size]
        Y_batch = Y[:batch_size]
        for x, y in zip(X_batch, Y_batch):
            derivatives = dervative(x, y, bo, b1)
            dbo += derivatives[0]
            db1 += derivatives[1]

        dbo = np.mean(dbo)
        db1 = np.mean(db1)

        vdw = (beta * vdw) + (1 - beta) * db1
        vdb = (beta * vdb) + (1 - beta) * dbo

        bo -= eta*vdb
        b1 -= eta*vdw

        print("At iteration no: ", t, " the loss of batch ", loss_function(bo, b1, Y_batch, X_batch), " absolute loss is ", loss_function(bo, b1, Y, X))

        bo_values.append(bo)
        b1_values.append(b1)

        if (abs(prev_bo - bo) < tolerance and abs(prev_b1 - b1) < tolerance):
            break;
        else:
            prev_bo= bo
            prev_b1 = b1

    return bo, b1

bo_final, b1_final = gradient_descent(x, y, 0.1, 0.1, epochs= 200, batch_size= 64)

print("The final weights are bo = ", np.round(bo_final, 3), " b1 = ", np.round(b1_final, 3))