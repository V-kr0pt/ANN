import numpy as np
import matplotlib.pyplot as plt
import Funcoes_de_Ativacao as fa

class Net:
    
    def __init__(self, neuronios):
        
        #inicializando os pesos e bias    
        self.W = [] #lista de arrays que irá conter os pesos aleatoriamente iniciados
        self.biases = [] #lista de arrays que irá conter os bias aleatoriamente iniciados

        #contabiliza o número de camadas retirando uma
        self.num_de_camadas = len(neuronios) - 1 
        
        #criando uma lista de 0 até o número de camadas
        self.camadas = list(range(self.num_de_camadas)) 
        
        #As conexões existem entre duas camadas: camada e camada+1.        
        for camada in self.camadas:
            #camada = 0 : vetor de entradas

            #inicializando os pesos randômicamente
            W = np.random.rand(neuronios[camada+1], neuronios[camada]) 
            W *= 0.1 #garantido pesos inicializados <= |0.1| 
            self.W.append(W) #salva na lista

            biases =  0.1 * np.random.rand(neuronios[camada+1]) #bias randômicos            
            self.biases.append(biases) #salva na lista
        

    def feedForward(self, x):
        self.x = np.array([x]) 

        #listas para deixar salvo as saídas e as somas das entradas dos neurônios
        self.output = [] 
        self.sum_fa = []
        
        #generalizando, a saída do neurônio de entrada é a prórpia entrada 
        self.output.append(self.x) 
                
        for camada in self.camadas:
             
            sum_fa =  np.dot(self.W[camada], self.output[camada]) + self.biases[camada]             
            output = fa.tansig(sum_fa)           
            
            self.sum_fa.append(sum_fa)
            self.output.append(output)        
        
        #retorna a saída da camada de saída
        return output
        

    def feedBackward(self, output, target, learning_rate):
        
        #a variável error da camada de saída 
        #é dada pela diferença entre o target e o a saída do neurônio  
              
        error = target - output    #erro da camada de saída        
        
        for camada in reversed(self.camadas):  
            dif_FA = fa.tansig(self.sum_fa[camada], diff=True)
            e = np.dot(error, dif_FA)                
            #atualização dos pesos 
            
            #transformando vetores de saída em matrizes coluna 
            output_rows = self.output[camada].shape[0]
            output_mat = self.output[camada].reshape((output_rows,1))

            #transformando os vetores "e", em matrizes coluna
            if e.shape:  #se for um vetor
                e_rows = e.shape[0] #recebe o número de linhas
                e_mat = e.reshape((e_rows,1)) #retorna uma matriz coluna
            else:
                e_mat = e

            self.W[camada] = \
                self.W[camada] + learning_rate * np.dot(e_mat, output_mat.T)
            

            #atualização dos biais da camada de saída
            self.biases[camada] = \
                self.biases[camada]  + learning_rate * np.sum(e, axis=0, keepdims=True)

            error = np.dot(e, self.W[camada]).sum() #erro da camada escondida          

        

    def train(self, X, y, learning_rate=0.1, goal= 1e-2, epochs = 10**3):


        #treinamento
        for _ in range(epochs):

            self.train_error = np.array([])

            #verifica os erros
            for i,x in enumerate(X): 
                
                output = NN.feedForward(x)
                
                tr_error = 1/2*(y[i]-output)**2                     
                
                self.train_error = np.append(self.train_error, tr_error)

            print(self.train_error.max())
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
        NN.train(Xtr, ytr, learning_rate=0.1, goal=1e-3, epochs=10**3)

        #Teste da RNA
        output = np.array([])

        for x in Xts:
            output = np.append(output, NN.feedForward(x))
        
        error_ts = abs(yts - output)
        #print(error_ts.max())
        
        if(error_ts.max() <= 0.02):
            NNBest = NN
            print("ACHOU!")
            break

    
    plt.stem(Xtr, ytr, 'b', markerfmt='bo', label='treino')
    plt.stem(Xts, yts,'g', markerfmt='go', label='teste')
    plt.stem(Xts, output, 'r', markerfmt='ro', label='output')
    plt.legend()
    plt.show()


