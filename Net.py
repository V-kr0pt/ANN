import numpy as np
import matplotlib.pyplot as plt
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
            #camada = 0 : vetor de entradas

            #inicializando os pesos randômicamente
            W = np.random.rand(neuronios[camada+1], neuronios[camada]) 
            W *= 0.1 #garantido pesos inicializados <= |0.1| 
            self.W.append(W) #salva na lista

            biases = np.random.rand(neuronios[camada+1]) #bias randômicos            
            self.biases.append(biases) #salva na lista
        

    def feedForward(self, x):
        self.x = np.array([x]) 

        #listas para deixar salvo as saídas e as somas das entradas dos neurônios
        self.output = [] 
        self.sum_fa = []
        
        #generalizando, a saída do neurônio de entrada é a prórpia entrada 
        self.output.append(self.x) 
                
        for conexao in self.conexoes:
             
            sum_fa =  np.dot(self.W[conexao], self.output[conexao]) + self.biases[conexao]             
            output = fa.ReLu(sum_fa)           
            
            self.sum_fa.append(sum_fa)
            self.output.append(output)        
        
        #retorna a saída da camada de saída
        return output
        

    def feedBackward(self, output, target, learning_rate):
        
        #invertendo a lista self.conexoes, para indicar o caminho de retropropagação
        b_conexoes = [(self.num_de_conexoes-1) - i for i in self.conexoes]

        #Foi necessário a retirada de 1 unidade de self.num_de_conexoes para que b_conexoes  
        #representasse uma lista contabilizando o número de conexões com índice iniciando em 0  

        #a variável error da camada de saída 
        #é dada pela diferença entre o target e o a saída do neurônio        
        error = target - output #erro da camada de saída        
        
        for conexao in b_conexoes:  
            dif_FA = fa.ReLu(self.sum_fa[conexao], diff=True)
            e = np.dot(error, dif_FA)                
        
            #atualização dos pesos 
            self.W[conexao] = \
                self.W[conexao] + learning_rate * np.dot(self.output[conexao+1], e)
            

            #atualização dos biais da camada de saída
            self.biases[conexao] = \
                self.biases[conexao]  + learning_rate * np.sum(e, axis=0, keepdims=True)

            error = np.dot(e, self.W[conexao]).sum() #erro da camada escondida          

        

    def train(self, X, y, learning_rate=0.1, goal= 1e-2, epochs = 10**3):


        #treinamento
        for _ in range(epochs):

            self.train_error = np.array([])

            #verifica os erros
            for i,x in enumerate(X): 
                
                output = NN.feedForward(x)
                
                tr_error = abs(y[i]-output)
                     
                
                self.train_error = np.append(self.train_error, tr_error)

            #se todos os erros forem menores ou iguais ao desejado, para o treinamento
            if(self.train_error.max() <= goal):
                break

            
            #c.c., faz o treinemento online:
            for i,x in enumerate(X): 
                output = NN.feedForward(x)
                NN.feedBackward(output, y[i], learning_rate)        



if __name__ == '__main__':
    
    t_d = np.linspace(0, 2*np.pi, 101) #ângulo em radianos  
    t = np.random.permutation(t_d) #permutando-o randomicamente
    y = np.sin(t)    #Função seno
    

    #treino 70% - teste 30%

    #Os vetores de treino:
    Xtr = t[0:70]
    ytr = y[0:70]

    #Os vetores de teste:
    Xts = t[71:101] 
    yts = y[71:101]

    for i in range(50):
        #Topologia da RNA
        NN = Net([1,10,5,1])        
        
        #Treina a RNA
        NN.train(Xtr, ytr, learning_rate=0.5, goal=1e-3, epochs=10**3)

        #Teste da RNA
        output = np.array([])

        for i,x in enumerate(Xts):
            output = np.append(output, NN.feedForward(x))
        
        error_ts = abs(yts - output)
        print(error_ts.max())
        
        if(error_ts.max() <= 0.02):
            NNBest = NN
            print("ACHOU!")
            break

    
    #plt.stem(Xtr, ytr)
    #plt.stem()
    #plt.stem(t_d, NNBest.feedForward(t_d))

        


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