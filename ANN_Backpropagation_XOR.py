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
