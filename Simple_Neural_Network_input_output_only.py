# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 19:51:09 2024

Simple artificial neural network (ANN) with single input and output layers based on 
"How to build a simple neural network in 9 lines of Python code" by Milo Spencer-Harper
https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1
    
ANN structure and features:        
    The input layers can be expanded to n nodes. The output layer has only 1 node
    Two activation functions are available: relu and sigmoid
    weight adjustment dw = Error * input * grad(activation function)


@author: Abdiel
"""
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt


class ANN():
    
    def __init__(self,x,y,n_out,activation,tol,max_iter,lr):
        
        #initialization network
        self.nodes_output = n_out       
        self.activation = activation      
        self.X =x.T
        self.Y =y.T
       
       # Initialization of solver parameters    
       
        #learning rate
        self.lr = lr       
        self.Norm_E0 = 1   
        
        #tolerance during traning process
        self.tol = tol       
        self.max_iter = max_iter       
        self.w0 = np.random.rand(self.nodes_output,self.X.shape[0])*0.1

  
    @classmethod
    
    # Two activation functions available as class methods
    # The user can select the one that it better fit its requirements
    # ReLU is adopted for efficiency   
    
    def Sigmoid_function(cls,y_node):        
        return 1/(1 + np.exp(-y_node)) 
    
    @classmethod
    def ReLU_function(cls,y_node):         
        return np.maximum(0,y_node)  
    
    # perform the sum(wi*xi) operation as a dot product
    def Compute_layer(self,w1):       
        return np.dot(w1,self.X) #+ self.bias1
            
    def Activation_function(self,y_node):      
        if self.activation == 'sigmoid':                  
            return ANN.Sigmoid_function(y_node)
        
        if self.activation == 'relu':             
            return ANN.ReLU_function(y_node)
    
    # gradient of activation function
    def Activation_function_derivative(self,y_node):           
        if self.activation == 'sigmoid':   
            return y_node * (1.- y_node)
        
        if self.activation == 'relu':             
            return np.ones((5,1))  
             
    def forward(self,w1):       
        y_node = self.Compute_layer(w1)        
        return self.Activation_function(y_node)    
    
    def Error_function(self,y_approx):        
        return y_approx - self.Y
    
    def Compute_norm_Error(self,E):       
        #vector norm of Error
        return LA.norm(E)       
        
    def Compute_weights_adjustment(self,Error,grad):        
        # x * E * grad(activation(y))
        return np.dot(self.X, Error * grad)
        
    def Train(self):        
        w1 = self.w0        
        Norm_Error = 1        
        iteration = 0
        
        # lists for plotting purposes
        iteration_vec = []; Error_vec = [];
        
        #forward and backward propagation during training
        while Norm_Error>self.tol and iteration < self.max_iter:
            
            #forward pass
            y_approx = self.forward(w1)  

            #Error calculations
            Error = self.Error_function(y_approx.T)
            Norm_Error = self.Compute_norm_Error(Error)
            
            #backward propagation + weights update
            grad = self.Activation_function_derivative(y_approx.T)
            dw = self.Compute_weights_adjustment(Error,grad)            
            w1 = w1 - self.lr * dw.T
            
            iteration += 1
            
            iteration_vec.append(iteration)
            Error_vec.append(Norm_Error)  
            
        return y_approx,Norm_Error,iteration_vec,Error_vec


    def Plot_Error(self,iteration_vec,Error_vec,activation):   
        
        plt.plot(iteration_vec, Error_vec, label =activation + ' function')       
        plt.ylabel('Norm of Error [-]')        
        plt.xlabel('Iteration No [-]')
        plt.legend(loc="upper right")
        plt.grid(True)

           
        
if __name__ == "__main__":

    #User data   
    
    #number of output nodes
    nodes_output = 1
    
    #training data
    x = np.array([[0, 0,1,0], [0.26, 0, 1,1], [1, 0, 1,0], [0.5,0, 1,1],[0.38,0, 1,1]])
    
    y = np.array([[0, 0.26, 1, 0.5,0.38]])
    
    #training output
    print('y_real as training vector is = ',y)

    #solver parameters
    lr = 0.01 
    
    max_iter = 1e5
    
    tol = 0.001
    
    for i in range(2):
        
        activation ={0:'relu',1:'sigmoid'}
        
        #Solution (using relu and sigmoid)
        simple_NN = ANN(x,y,nodes_output,activation[i],tol,max_iter,lr)   
       
        args = simple_NN.Train()
       
        #plot training error
        simple_NN.Plot_Error(args[2],args[3],activation[i])
    
        print('y_approx obtained after training' + ' with ', activation[i],'fuction' 'is = ',np.round(args[0],4))
        print('Norm of error is ', '{:.4E}'.format(args[1]))
    






       

