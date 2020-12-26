import numpy as np
import Funcoes_de_Ativacao as fa

class Net:
    
    def __init__(self, neuronios):
        
        #inicializando os pesos e bias    
        self.W = [] #lista de arrays contendo os pesos aleatoriamente iniciados
        self.biases = [] #lista de arrays contendo os bias aleatoriamente iniciados

        #contabiliza o número de conexões que existem na rede 
        self.num_de_conexoes = len(neuronios) - 1 
        
        #criando uma lista de 0 até o número de conexões
        self.conexoes = list(range(self.num_de_conexoes)) 
        
        #As conexões existem entre duas camadas, camada e camada+1.        
        for camada in self.conexoes:
         
            #pesos randômicos
            W = np.random.rand(neuronios[camada+1], neuronios[camada]) 
            W *= 0.1 #garantido pesos inicializados <= |0.1| 
            self.W.append(W) #salva na lista

            biases = np.random.rand(neuronios[camada+1]) #bias randômicos            
            self.biases.append(biases) #salva na lista
        

    def feedForward(self, X):
        self.X = X 

        #listas para deixar salvo as saídas e as somas das entradas dos neurônios
        self.output = [] 
        self.sum_fa = []
        
        #generalizando, a saída do neurônio de entrada é a prórpia entrada 
        self.output.append(self.X) 
                
        for conexao in self.conexoes:
                    
            sum_fa =  np.dot(self.output[conexao], self.W[conexao].T) + self.biases[conexao] 
            output = fa.ReLu(sum_fa)           

            self.sum_fa.append(sum_fa)
            self.output.append(output)        
        
        #retorna a saída da camada de saída
        return output
        
    def feedBackward(self, target, learning_rate):
        
        #invertendo a lista self.conexoes, para indicar o caminho de retropropagação
        b_conexoes = [self.num_de_conexoes-i-1 for i in self.conexoes]

        #Foi necessário a retirada de 1 unidade de self.num_de_conexoes para que b_conexoes  
        #representasse uma lista contabilizando o número de conexões com índice iniciando em 0  

        #a camada de saída é a camada posterior a última conexão        
        camada_de_saida = b_conexoes[0] + 1

        #a variável error da camada de saída 
        #é dada pela diferença entre o target e o a saída do neurônio       
        error = target - self.output[camada_de_saida] #erro da camada de saída        
        
        for conexao in b_conexoes:  
            dif_FA = fa.ReLu(self.sum_fa[conexao], diff=True)
            e = np.dot(error, dif_FA)
            print(error.shape)
            print(dif_FA.shape)
            print(f"{self.output[conexao].shape} \n")          

            
            #atualização dos pesos 
            self.W[conexao] = \
                self.W[conexao] + learning_rate * np.dot(self.output[conexao+1], e)
            

            #atualização dos biais da camada de saída
            self.biases[conexao] = \
                self.biases[conexao]  + learning_rate * np.sum(e, axis=0, keepdims=True)

            error = np.dot(e, self.W[conexao]).sum() #erro da camada escondida

            
           

        print("Backward sucesso!")
            

            

            
            
            


if __name__ == '__main__':
    NN = Net([2,3,3,1])
    X = np.array([[0,0],[0,1],[1,0],[1,1]])
    y = np.array([[0],[1],[1],[0]])
    for i,x in enumerate(X): 
        print(x)       
        output = NN.feedForward(x)
        NN.feedBackward(y[i], 0.1)


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