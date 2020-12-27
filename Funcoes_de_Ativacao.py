import numpy as np

def ReLu(x, diff=False):
    if diff: #retorna a derivada da função ReLu aplicada em x
        output = np.where(x <= 0, 0, 1) #retorna 0 aonde x for <=0 e 1 do contrário
    
    else: #retorna a função ReLu aplicada em x
        output = np.where(x <= 0, 0, x) #retorna 0 aonde x for <=0 e x do contrário
    
    return output

def tansig(x, diff=False):
    if diff:
        return 1/(np.cosh(x)**2)
    else:
        return np.tanh(x) 
                
if __name__ == "__main__":
    x = np.array([[-2,0,-1,2,1,0,5,-10], [15,0,-1,2,1,2,5,-15]])
    print(f"A entrada:\n {x}")
    print(f"A função aplicada na entrada:\n {ReLu(x)}") 
    print(f"A derivada da função aplicada na entrada:\n {ReLu(x, diff=True)}")