"""
Esse código tem o objetivo de elucidar o algoritmo de Backpropagation, de forma a 
criar uma RNA capaz de se aproximar ao comportamento de uma porta lógica XOR  
"""

import numpy as np
import matplotlib.pyplot as plt

#entrada
X = np.array([[0,0],[0,1],[1,0],[1,1]])
#saída esperada
y = np.array([[0],[1],[1],[0]]) 

#definindo a função de ativação
def sigmoid(x, diff=False):
    
    if diff:
        #derivada da função sigmoid
        return sigmoid(x) * (1 - sigmoid(x))
    else:
        #função sigmoid
        return 1/(1 + np.exp(-x))

class Neural():
    
    def __init__(self):
        self.inputLayerNeurons, self.hiddenLayerNeurons, self.outputLayerNeurons = 2, 2, 1

        #inicializando os pesos       
        self.W_hidden = np.random.rand(self.inputLayerNeurons, self.hiddenLayerNeurons)
        self.W_output = np.random.rand(self.hiddenLayerNeurons, self.outputLayerNeurons)
        
        #inicializando os biais
        self.bias_hidden = np.random.rand(1, self.hiddenLayerNeurons)
        self.bias_output = np.random.rand(1, self.outputLayerNeurons)

    #definindo feed forward propagation
    def feedForward(self, x):
        
        #camada escondida
        self.hidden_net = np.dot(x, self.W_hidden) + self.bias_hidden # 4x2 * 2x2  + 1x2 -> 4x2
        self.hidden_output = sigmoid(self.hidden_net) # 4x2

        #camada de saída
        self.output_net = np.dot(self.hidden_output, self.W_output) + self.bias_output # 1x2 * 2x1 -> 1x1
        output = sigmoid(self.output_net) # 1x1

        return output

    #definindo feed backward propagation
    def feedBackward(self, x, learning_rate, expected_out, output):
        #erro da camada de saída
        self.error = (expected_out - output) # 1x1
        self.delta_output = self.error * sigmoid(self.output_net, diff=True) #1x1
        
        #erro da camada escondida
        self.hidden_error = np.dot(self.delta_output, self.W_output.T) # 1x1 * 1x2 -> 1x2
        self.delta_hidden = self.hidden_error * sigmoid(self.hidden_net, diff=True) #1x2

        #atualizando os pesos
        self.W_output += learning_rate * np.dot(self.hidden_output.T, self.delta_output) #2x1 * 1x1 -> 2x1 
        self.W_hidden += learning_rate * np.dot(x.T, self.delta_hidden) #2x1 * 1x2 -> 2x2

        #atualizando os bias
        self.bias_output += learning_rate * np.sum(self.delta_output, axis=0, keepdims=True)  
        self.bias_hidden += learning_rate * np.sum(self.delta_hidden, axis=0, keepdims=True)
        
    #definindo treinamento
    def train(self, minimum_error, learning_rate, expected_out, X):
        #inicializando as épocas e os erros:
        epochs = 0
        error = np.ones(expected_out.shape) 
        error_epochs = []
        #se qualquer um dos erros for maior ou igual ao erro mínimo
        while any(error >= minimum_error) :
            #contador da época
            epochs+=1    
            
            #ajuste de pesos
            self.output = self.feedForward(X)
            self.feedBackward(X, learning_rate, expected_out, self.output)

            #cálculo do erro        
            error = abs(self.feedForward(X) - expected_out) 

            #erro absoluto médio
            error_epochs.append(error.sum()/4)        
            
            
            print(f"época: {epochs}\nerro: \n{error}\n")

        #plot do gráfico erro abs médio vs época
        plt.plot(range(epochs), error_epochs) 
        plt.xlabel("Época", size=10)
        plt.ylabel("Erro", size=10)
        plt.show()

        #retorno dos dados da RNA treinada
        return epochs, error, self.W_hidden, self.W_output, self.bias_hidden, self.bias_output

    
            
            

if __name__ == '__main__':
    NN = Neural()
    #determinado a taxa de aprendizagem
    learning_rate = 0.9
    minimum_error = 0.1
   
     #treinando a rede:
    epoch, error, W_escondido, W_saida, bias_escondido, bias_saida = NN.train(minimum_error, learning_rate, y, X)
    print(f"a época que atendeu ao requisitado foi: {epoch}\no erro: \n{error}\n")
    print(f"Os pesos da camada escondida: \n{W_escondido}\n, Os pesos da camada de saída:\n {W_saida}\n")
    print(f"O bias da camada escondida: \n{bias_escondido}\n, O biais da camada de saída:  \n{bias_saida}\n")
