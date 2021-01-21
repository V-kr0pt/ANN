import numpy as np
import matplotlib.pyplot as plt
from Net import ANN 


def criar_figuras():
    
    #figura para apresentar a melhor curva de treinamento e validação
    ax1.set_xlabel("Época", fontsize=20)
    ax1.set_ylabel("Erro Médio Quadrático", fontsize=20, rotation=90)
    ax1.tick_params(labelsize=20)
    ax1.tick_params(labelsize=20)    

    #figura para apresentar o erro de teste    
    ax2.set_xlabel("Tentativas", fontsize=20)
    ax2.set_ylabel("Erro Absoluto", fontsize=20, rotation=90)
    ax2.tick_params(labelsize=20)
    ax2.tick_params(labelsize=20)
    

def plot_figuras():

    #plot do gráfico do melhor resultado obtido
    ax1.plot(NNBest.train_mse, 'b', label = f"erro do treinamento {n+1}") 
    ax1.plot(NNBest.validation_mse, 'r', label = f"erro de validação {n+1}")
    ax1.legend(fontsize = 20)
    ax1.grid() 

    #plot do gráfico do erro de teste
    ax2.plot(array_test_error, 'b', label = f"erro de teste")   
    ax2.legend(fontsize = 20)
    ax2.grid()

    #plot do gráfico do seno para comparação
    plt.stem(Xtr, ytr, 'b', markerfmt='bo', label='treino')
    plt.stem(Xts, yts,'g', markerfmt='go', label='teste')
    plt.stem(Xts, output, 'r', markerfmt='ro', label='output')
    plt.legend()


#Criando o banco de dados: função seno
t_d = np.linspace(0, 2*np.pi, 100) #ângulo em radianos  
t = np.random.permutation(t_d) #permutando-o randomicamente
y = np.sin(t)    #Função seno


#Divisão dos dados em treino e teste:
#treino 70% - teste 30%

#Os vetores de treino:
Xtr = t[0:70]
ytr = y[0:70]

#Os vetores de teste:
Xts = t[70:100] 
yts = y[70:100]


#Inicialização dos gráficos:
fig1, ax1 = plt.subplots(figsize=(18,8))
fig2, ax2 = plt.subplots(figsize=(18,8)) 
fig3, ax3 = plt.subplots(figsize=(18,8))
criar_figuras()
  

#inicializando o array de erro mínimo encontrado
array_test_error = np.array([2]) #o erro inicial será "2"

#selecionando a quantidade de testes desejada
test_count = 3

for i in range(test_count):

    #Topologia da RNA
    NN = ANN([1,3,3,1])        
    
    #Treinando a RNA
    NN.train(Xtr, ytr, learning_rate=1e-3, goal=1e-2, epochs=10**3, validation_batch=0.3)    
    
    #Teste da RNA

    #inicializando o vetor de saídas
    output = np.array([])

    #realiza o feedForward com os dados de teste salvando no vetor "output"
    for x in Xts:
        output = np.append(output, NN.feedForward(x))
    
    #Cálculo do erro teste (erro absoluto)  
    test_error = abs(yts - output)      
    
    #verificando se o erro do teste atual é menor que o menor erro de teste já atingido 
    if(test_error.max() < array_test_error[-1]):    

        #salvando a melhor rede encontrada até o momento 
        NNBest = NN

        #salvando a informação de qual teste foi o melhor
        n = i        

        #verificando se o erro é menor que a tolerância desejada
        if(array_test_error[-1] <= 0.02):
            print("ACHOU!")            
            break
    
    #Salvando o erro de teste no vetor
    array_test_error = np.append(array_test_error, test_error.max())


#plotando os gráficos
plot_figuras()
plt.show()

