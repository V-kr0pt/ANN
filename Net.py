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
              
        error = target - output       #erro da camada de saída        
        
        
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

                  
            
        

    def train(self, X, y, learning_rate=1e-3, goal= 1e-2, epochs = 10**3, validation_batch = 0.25):        
        
        #No primeiro momento realizando a divisão para o caso específico dos dados seno
        #divisão 25% validação e 75% treino

        data_length = int( validation_batch * len(X) )  

        X_validation = X[0:data_length]
        y_validation = y[0:data_length]     
        X_train = X[data_length+1:]
        y_train =  y[data_length+1:]        
        
        
        #array que irá conter os erros máximos do processo de treino
        self.train_error_max = np.array([])

        #array que irá conter os erros máximos do processo de validação
        self.validation_error_max = np.array([])

        #inicialização do contador de épocas para validação
        count = 0

        #treinamento
        for epoca in range(epochs):
            
            #array que irá armazenar os erros de treino referentes a cada saída
            self.train_error = np.array([])
            
            #array que irá armazenar os erros de validação referentes a cada saída
            self.validation_error = np.array([])

            #Processo de atualização dos pesos
            for i,x in enumerate(X_train): 
                
                #saída da rede neural para as entradas de treino
                output = self.feedForward(x)
                
                #cálculo do erro de treinamento (MSE)
                train_error = 1/2 * (y_train[i] - output) ** 2        
                
                #salvando o erro no array de erro de treinamento
                self.train_error = np.append(self.train_error, train_error)               

                #FeedBackward propagation
                self.feedBackward(output, y_train[i], learning_rate)

            
            #Processo de validação
            for i,x in enumerate(X_validation):

                #saída da rede neural para as entradas de validação
                output = self.feedForward(x)

                #cálculo do erro de validação (MSE)
                validation_error = 1/2 * (y_validation[i] - output) ** 2 

                #salvando o erro no array de erro de validação
                self.validation_error = np.append(self.validation_error, validation_error)   


            #criando arrays com os erros máximos para serem plotados e avaliados
            self.train_error_max = np.append(self.train_error_max, self.train_error.max())
            self.validation_error_max = np.append(self.validation_error_max, self.validation_error.max())


            #Critérios de parada:

            #parada se o erro do treino for menor ou igual ao desejado
            if(self.train_error_max[-1] <= goal):
                    print("Treinamento parado pois o objetivo foi atingido!")
                    break            
            
            
            if(epoca > 0): 
                ''' 
                #parada se a taxa de variação do erro de treino for maior que -0.001     
                train_error_rate = self.train_error_max[-1] - self.train_error_max[-2]      
                
                
                if(train_error_rate > -1e-5):
                    print("Treinamento parado pela variação do erro de treinamento")
                    break
                '''
                
                #parada se a taxa de variação do erro de validação for não negativa por 100 épocas
                validation_error_rate = self.validation_error_max[-1] - self.validation_error_max[-2] 
                
                if(validation_error_rate >= 0):                                      
                    count+=1
                    
                    if(count == 10):
                        #retorna a melhor configuração da rede
                        self.W = best_weigths
                        self.biases = best_biases
                        print("Treinamento parado pelo erro de validação")
                        break
                
                else:
                    #salva a melhor configuração da rede
                    best_weigths = self.W
                    best_biases = self.biases             
                    count = 0
                print(f"erro de treinamento: {self.train_error_max[-1]}\nerro de validação: {self.validation_error_max[-1]}")
        
        print(f"\nmelhor erro de treinamento: {self.train_error_max.min()}\nmelhor erro de validação: {self.validation_error_max.min()}")

                
                          




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

    for i in range(2):
        #Topologia da RNA
        NN = Net([1,3,3,1])        
        
        #Treina a RNA
        NN.train(Xtr, ytr, learning_rate=1e-3, goal=1e-2, epochs=10**4, validation_batch=0.3)
        
        #plot do gráfico dos erros de treinamento e validação

        #vetor respectivo ao tamanho do vetor de erros de treino
        train_length = np.arange(len(NN.train_error_max))

        #vetor respectivo ao tamanho do vetor de erros de validação
        validation_lenght = np.arange(len(NN.validation_error_max))

        plt.plot(train_length, NN.train_error_max, label = f"treinamento {i}")
        plt.plot(validation_lenght, NN.validation_error_max, label = f"validação {i}")



    plt.xlabel("epochs")
    plt.ylabel("MSE")    
    plt.legend()
    plt.show()

        
        

"""         #Teste da RNA
        output = np.array([])

        for x in Xts:
            output = np.append(output, NN.feedForward(x))
        
        error_ts = abs(yts - output)
        #print(error_ts.max())
        
        if(error_ts.max() <= 0.06):
            NNBest = NN
            print("ACHOU!")
            break

    
    plt.stem(Xtr, ytr, 'b', markerfmt='bo', label='treino')
    plt.stem(Xts, yts,'g', markerfmt='go', label='teste')
    plt.stem(Xts, output, 'r', markerfmt='ro', label='output')
    plt.legend()
    plt.show()
 """

