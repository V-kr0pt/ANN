"""
Esse código é uma tentativa de fazer um multilayer perceptron que implementa a porta lógica XOR utilizando
o algoritmo de backpropagation como algoritmo de aprendizagem
"""

import numpy as np

#Input dataset
X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([[0],[1],[1],[0]]) #saída esperada

#definindo a função de ativação
def sigmoid(x, diff=False):
    
    if diff:
        #derivada da função sigmoid
        return x * (1 - x)
    else:
        #função sigmoid
        return 1/(1 + np.exp(-x))

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

class Neural():
    #inicializando os pesos
    def __init__(self):
        self.W_hidden = np.random.uniform(low=-0.1, high=0.1, size= (inputLayerNeurons, hiddenLayerNeurons)) #2x2
        self.W_output = np.random.uniform(low=-0.1, high=0.1, size=(hiddenLayerNeurons, outputLayerNeurons)) #2x1

    #definindo feed forward propagation
    def feedForward(self, x):
        #camada escondida
        self.hidden_net = np.dot(x, self.W_hidden) # 1x2 * 2x2 
        self.hidden_output = sigmoid(self.hidden_net) # 1x2

        #camada de saída
        self.output_net = np.dot(self.hidden_output.T, self.W_output) # 1x2 * 2x1
        output = sigmoid(self.output_net) # 1x1

        return output

    #definindo feed backward propagation
    def feedBackward(self, x, learning_rate, expected_out, output):
        #erro da camada de saída
        self.error = (expected_out - output) # 1x1
        self.delta_output = self.error * sigmoid(output, diff=True) #1x1
        
        #erro da camada escondida
        self.hidden_error = np.dot(self.delta_output, self.W_output.T) # 1x1 * 1x2 
        self.delta_hidden = self.hidden_error * sigmoid(self.hidden_output, diff=True) #1x2

        #atualizando os pesos
        self.W_output += learning_rate * np.dot(self.hidden_output.T, self.delta_output) # 2x1 * 1x1 
        self.W_hidden += learning_rate * np.dot(x.T, self.delta_hidden) #2x1 * 1x2 

    def train(self, epochs, learning_rate, expected_out, X, y):
        for i,x in enumerate(X):
            for _ in epochs:
                self.output = self.feedForward(x)
                self.feedBackward(x, learning_rate, expected_out, y[i])


if __name__ == '__main__':
    NN = Neural()
    output = NN.feedForward(X[0])
    print (output)