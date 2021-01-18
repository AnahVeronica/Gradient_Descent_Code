import numpy as np
from sklearn.utils import shuffle

#Data points
x = np.linspace(0,5, 500)

#True Function
bo = 2
b1 = 3.14
y = bo + b1*x

def shuffled_data(x):
    x = shuffle(x)
    y = bo + b1*x
    return x, y

#Loss function to compute error
def loss_function(bo, b1, y, x):
    squared_error = (y - (bo + b1*x))**2
    return np.mean(squared_error)

#Derivative of the loss function with respect to the weights bo and b1
def dervative(x, y, bo, b1):
    bo_derv = 2*(-y + bo + b1*x)
    b1_derv = 2*x*(-y + bo + b1*x)
    return bo_derv, b1_derv

def square(list): #To return square of sum of derivatives
    if list != []:
        return sum([i ** 2 for i in list])
    else:
        return 0


def gradient_descent(X, Y, bo, b1, eta=0.0001, tolerance=0.0001, epochs = 1, epsilon = 0.01, batch_size = 16):
    prev_bo = 0
    prev_b1 = 0

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

        eta_delta = eta / np.sqrt(square(b1_values) + epsilon)


        bo -= eta_delta*dbo
        b1 -= eta_delta*db1

        print("At iteration no: ", t, " the loss of batch ", loss_function(bo, b1, Y_batch, X_batch), " absolute loss is ", loss_function(bo, b1, Y, X))

        bo_values.append(bo)
        b1_values.append(b1)

        if (abs(prev_bo - bo) < tolerance and abs(prev_b1 - b1) < tolerance):
            break;
        else:
            prev_bo= bo
            prev_b1 = b1

    return bo, b1

bo_final, b1_final = gradient_descent(x, y, np.random.randn(), np.random.randn(), epochs= 200, batch_size= 64)

print("The final weights are bo = ", np.round(bo_final, 3), " b1 = ", np.round(b1_final, 3))