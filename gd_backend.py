import numpy as np 
import matplotlib.pyplot as plt
X= [
    [2,60,45],
    [3,65,50],
    [4,70,55],
    [5,75,60],
    [6,80,65],
    [7,85,70],
    [8,90,75],
    
]

Y=[0,0,0,1,1,1,1]  
#input -> hidden (3*2)

w1= 0.1 ; w2= -0.2
w3= 0.4 ; w4= 0.2
w5= -0.5 ; w6= 0.1

bh1=0.1
bh2= -0.1

#hidden -> output (2*1)
w7= 0.3 ; w8= -0.3
bo= 0.2
lr= 0.05
epochs= 500

for epoch in range(epochs):
    total_error=0
    for i in range(len(X)):
        x1= X[i][0]
        x2= X[i][1] 
        x3= X[i][2]
        
        zh1= w1*x1 + w3*x2 + w5*x3 + bh1
        zh2= w2*x1 + w4*x2 + w6*x3 + bh2
        
        h1= 1/(1 + (2.71828)**(-zh1))
        h2= 1/(1 + (2.71828)**(-zh2))
        z0= w7*h1 + w8*h2 + bo
        o= 1/(1 + (2.71828)**(-z0))
        #error
        error= (Y[i] - o)**2
        total_error = total_error + (error* error)
        
        #backpropagation
        delta_o= error * o * (1 - o)
        delta_h1= delta_o * w7 * h1 * (1 - h1)
        delta_h2= delta_o * w8 * h2 * (1 - h2)
        #update weights and biases
        w7= w7 + lr * delta_o * h1
        w8= w8 + lr * delta_o * h2
        bo= bo + lr * delta_o
        w1= w1 + lr * delta_h1 * x1
        w3= w3 + lr * delta_h1 * x2
        w5= w5 + lr * delta_h1 * x3
        bh1= bh1 + lr * delta_h1
        w2= w2 + lr * delta_h2 * x1
        w4= w4 + lr * delta_h2 * x2
        w6= w6 + lr * delta_h2 * x3
        bh2= bh2 + lr * delta_h2
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Total Error: {total_error:.4f} output: {o:.4f}")