#Import the libraries
import numpy as np

#Data points
x = np.linspace(0,5, 500)

#True Function
bo = 2
b1 = 3.14
y = bo + b1*x

#Loss function to compute error
def loss_function(bo, b1, y, x):
    squared_error = (y - (bo + b1*x))**2
    return np.mean(squared_error)

#Derivative of the loss function with respect to the weights bo and b1
def dervative(x, y, bo, b1):
    bo_derv = 2*(-y + bo + b1*x)
    b1_derv = 2*x*(-y + bo + b1*x)
    return bo_derv, b1_derv

def gradient_descent(X, Y, bo, b1, eta=0.01, tolerance=0.001, epochs = 1):
    prev_bo = 0
    prev_b1 = 0

    t = 0
    n = 1

    bo_values = [bo]
    b1_values = [b1]

    for i in range(epochs):
        dbo = 0
        db1 = 0
        t += 1
        for x, y in zip(X, Y):
            derivatives = dervative(x, y, bo, b1)
            dbo += derivatives[0]
            db1 += derivatives[1]

            bo -= eta*dbo
            b1 -= eta*db1

            print("The ", n, "th loss is ", loss_function(bo, b1, y, x))
            n += 1
            bo_values.append(bo)
            b1_values.append(b1)
            if (abs(prev_bo - bo) < tolerance and abs(prev_b1 - b1) < tolerance):
                break;
            else:
                prev_bo= bo
                prev_b1 = b1

    return bo, b1

bo_final, b1_final = gradient_descent(x, y, 0.1, 0.1, epochs= 200)

print("The final weights are bo = ", np.round(bo_final, 3), " b1 = ", np.round(b1_final, 3))