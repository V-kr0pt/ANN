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

class Neural():
    
    def __init__(self):
        self.inputLayerNeurons, self.hiddenLayerNeurons, self.outputLayerNeurons = 2, 2, 1

        #inicializando os pesos
        self.W_hidden = np.random.uniform\
            (low=-0.1, high=0.1, size=(self.inputLayerNeurons, self.hiddenLayerNeurons)) #2x2
        self.W_output = np.random.uniform\
            (low=-0.1, high=0.1, size=(self.hiddenLayerNeurons, self.outputLayerNeurons)) #2x1
        
        #inicializando os biais
        self.bias_hidden = np.random.uniform(size=(1, self.hiddenLayerNeurons))
        self.bias_output = np.random.uniform(size=(1, self.outputLayerNeurons))

    #definindo feed forward propagation
    def feedForward(self, x):
        
        #camada escondida
        self.hidden_net = np.dot(x.T, self.W_hidden) # 1x2 * 2x2 -> 1x2
        self.hidden_output = sigmoid(self.hidden_net) # 1x2

        #camada de saída
        self.output_net = np.dot(self.hidden_output, self.W_output) # 1x2 * 2x1 -> 1x1
        output = sigmoid(self.output_net) # 1x1

        return output

    #definindo feed backward propagation
    def feedBackward(self, x, learning_rate, expected_out, output):
        #erro da camada de saída
        self.error = (expected_out - output) # 1x1
        self.delta_output = self.error * sigmoid(output, diff=True) #1x1
        
        #erro da camada escondida
        self.hidden_error = np.dot(self.delta_output, self.W_output.T) # 1x1 * 1x2 -> 1x2
        self.delta_hidden = self.hidden_error * sigmoid(self.hidden_output, diff=True) #1x2

        #atualizando os pesos
        self.W_output += learning_rate * np.dot(self.hidden_output.T, self.delta_output) #2x1 * 1x1 -> 2x1 
        self.W_hidden += learning_rate * np.dot(x, self.delta_hidden) #2x1 * 1x2 -> 2x2

    def train(self, epochs, learning_rate, expected_out, X):
        for i, x in enumerate(X):
            for _ in range(epochs):
                self.output = self.feedForward(x.reshape(2,1))
                self.feedBackward(x.reshape(2,1), learning_rate, expected_out[i], self.output)


if __name__ == '__main__':
    NN = Neural()
    #treinando a rede:
    epochs = 15000
    learning_rate = 0.3
    NN.train(epochs, learning_rate, y, X) 

    #avaliando a rede treinada:
    output = np.zeros((4,1))
    for i, x in enumerate(X):
        output[i] = NN.feedForward(x.reshape(2,1))
    print (output)    