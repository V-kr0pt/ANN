def ReLu(x, diff=False):
    if x <= 0: 
            return 0 
    else:
        if diff:
            return 1 
        else:
            return x