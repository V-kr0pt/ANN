import numpy as np
import matplotlib.pyplot as plt
from Net import ANN 


t_d = np.linspace(0, 2*np.pi, 100) #ângulo em radianos  
t = np.random.permutation(t_d) #permutando-o randomicamente
y = np.sin(t)    #Função seno

#treino 70% - teste 30%
#Os vetores de treino:
Xtr = t[0:70]
ytr = y[0:70]
#Os vetores de teste:
Xts = t[70:100] 
yts = y[70:100]
#Figura para o gráfico dos erros de treinamento e validação:
fig1, ax1 = plt.subplots(figsize=(18,8))
ax1.set_xlabel("Época", fontsize=20)
ax1.set_ylabel("Erro Médio Quadrático", fontsize=20, rotation=90)
ax1.tick_params(labelsize=20)
ax1.tick_params(labelsize=20)
#figura para apresentar a melhor curva de treinamento e validação
fig2, ax2 = plt.subplots(figsize=(18,8)) 
ax2.set_xlabel("Época", fontsize=20)
ax2.set_ylabel("Erro Médio Quadrático", fontsize=20, rotation=90)
ax2.tick_params(labelsize=20)
ax2.tick_params(labelsize=20)
#figura para apresentar o erro de teste
fig3, ax3 = plt.subplots(figsize=(18,8)) 
ax3.set_xlabel("Tentativas", fontsize=20)
ax3.set_ylabel("Erro Absoluto", fontsize=20, rotation=90)
ax3.tick_params(labelsize=20)
ax3.tick_params(labelsize=20)

#inicializando o array de erro mínimo encontrado
array_test_error = np.array([2])
for i in range(50):
    #Topologia da RNA
    NN = ANN([1,3,3,1])        
    
    #Treina a RNA
    NN.train(Xtr, ytr, learning_rate=1e-3, goal=1e-2, epochs=10**3, validation_batch=0.3)
    #vetor respectivo ao tamanho do vetor de erros de treino
    train_mse_length = np.arange(len(NN.train_mse))
    #vetor respectivo ao tamanho do vetor de erros de validação
    validation_mse_length = np.arange(len(NN.validation_mse))
    ax1.plot(train_mse_length, NN.train_mse, label = f"erro do treinamento {i+1}")
    ax1.plot(validation_mse_length, NN.validation_mse, label = f"erro de validação {i+1}")        
    
    #Teste da RNA
    output = np.array([])
    #realiza feedForward com os dados de teste
    for x in Xts:
        output = np.append(output, NN.feedForward(x))
    
    #Cálculo do erro absoluto de teste
    test_error = abs(yts - output)      
    
    #verifica se esse erro é menor que o erro mínimo já atingido 
    if(test_error.max() < array_test_error[-1]):            
        #salvando a rede 
        NNBest = NN
        #salvando dados para o plot
        n = i
        train_mse_length_best = train_mse_length
        validation_mse_length_best = validation_mse_length
        #verifica se o erro é menor que a tolerância
        if(array_test_error[-1] <= 0.02):
            print("ACHOU!")            
            break
    
    #adição do erro ao array de erros de teste
    array_test_error = np.append(array_test_error, test_error.max())

#plot do gráfico de todas as curvas de treinamento e validação 
ax1.grid()
ax1.legend(fontsize = 20)
#plot do gráfico do melhor resultado obtido
ax2.plot(train_mse_length_best, NNBest.train_mse, 'b', label = f"erro do treinamento {n+1}") 
ax2.plot(validation_mse_length_best, NNBest.validation_mse, 'r', label = f"erro de validação {n+1}")    
ax2.grid()
ax2.legend(fontsize = 20)
#plot do gráfico do erro de teste
ax3.plot(array_test_error, 'b', label = f"erro de teste")   
ax3.grid()
ax3.legend(fontsize = 20)
'''
plt.stem(Xtr, ytr, 'b', markerfmt='bo', label='treino')
plt.stem(Xts, yts,'g', markerfmt='go', label='teste')
plt.stem(Xts, output, 'r', markerfmt='ro', label='output')
plt.legend()
'''
plt.show()

