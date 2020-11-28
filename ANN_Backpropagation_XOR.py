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
        return x * (1 - x)
    else:
        return 1/(1 + np.exp(-x))

inputLayerNeurons, hiddenLayerNeurons, outputLayerNeurons = 2, 2, 1

#inicializando os pesos
W_hidden = np.random.uniform(low=-0.1, high=0.1, size= (inputLayerNeurons, hiddenLayerNeurons))
W_output = np.random.uniform(low=-0.1, high=0.1, size=(hiddenLayerNeurons, outputLayerNeurons))

#definindo feed forward propagation
def feedforward(x):
    #camada escondida
    hidden_net = np.dot(x, W_hidden)
    hidden_output = sigmoid(hidden_net)

    #camada de saída
    output_net = np.dot(hidden_output.T, W_output)
    output = sigmoid(output_net)