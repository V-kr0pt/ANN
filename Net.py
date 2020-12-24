import numpy as np
import Funcoes_de_Ativacao as fa

class Net:
    
    def __init__(self, neuronios):
        
        #inicializando os pesos (lista de pesos)   
        self.W = [] #lista de arrays contendo os pesos aleatoriamente iniciados
        self.num_de_camadas = len(neuronios-1)
        for i in range(self.num_de_camadas):
            W = np.random.rand(neuronios[i+1], neuronios[i]) #pesos randômicos
            #biases = np.random.rand(neuronios[i]) #bias randômicos
            W *= 0.1 #garantido pesos inicializados <= |0.1| 
            self.W.append(W)

   


if __name__ == '__main__':
    NN = Net([1,5,5,1])


"""
    #definindo feed forward propagation
    def feedForward(self, x):
        
        #camada escondida
        self.hidden_net = np.dot(x, self.W_hidden) + self.bias_hidden 
        self.hidden_output = sigmoid(self.hidden_net) 

        #camada de saída
        self.output_net = np.dot(self.hidden_output, self.W_output) + self.bias_output 
        output = sigmoid(self.output_net) 

        return output

    #definindo feed backward propagation
    def feedBackward(self, x, learning_rate, expected_out, output):
        #erro da camada de saída
        self.error = (expected_out - output) 
        self.delta_output = self.error * sigmoid(self.output_net, diff=True) 
        
        #erro da camada escondida
        self.hidden_error = np.dot(self.delta_output, self.W_output.T) 
        self.delta_hidden = self.hidden_error * sigmoid(self.hidden_net, diff=True) 

        #atualizando os pesos
        self.W_output += learning_rate * np.dot(self.hidden_output.T, self.delta_output) 
        self.W_hidden += learning_rate * np.dot(x.T, self.delta_hidden) 

        #atualizando os bias
        self.bias_output += learning_rate * np.sum(self.delta_output, axis=0, keepdims=True)  
        self.bias_hidden += learning_rate * np.sum(self.delta_hidden, axis=0, keepdims=True) """