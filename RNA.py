import numpy as np

class Neural:
    
    def __init__(self, X, y, hiddenLayerNeurons):

        #inicializando os pesos   

        self.W = []
        for i, qtd_de_neuronios_na_camada in enumerate(hiddenLayerNeurons):
            if i == 0:
                #pesos da primeira camada escondida
                self.W.append(np.random.rand(X.shape[0], qtd_de_neuronios_na_camada))
            else:
                #pesos das camadas escondidas intermediarias
                self.W.append(np.random.rand(hiddenLayerNeurons[i-1], qtd_de_neuronios_na_camada))
        
        #pesos da camada de saída
        self.W.append(np.random.rand(y.shape[0],hiddenLayerNeurons[-1]))


            
        #self.W_hidden = np.random.rand(self.inputLayerNeurons, self.hiddenLayerNeurons)
        #self.W_output = np.random.rand(self.hiddenLayerNeurons, self.outputLayerNeurons)
        
        #inicializando os biais
        #self.bias_hidden = np.random.rand(1, self.hiddenLayerNeurons)
        #self.bias_output = np.random.rand(1, self.outputLayerNeurons)





"""     #definindo feed forward propagation
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