import math

def linear(x):
    if x >= 0.5:
        return 1
    elif x >= -0.5:
        return x + 0.5
    else:
        return 0

def sigmoid(x, alpha = 1):
    return 1 / (1 + math.exp(-(alpha * x)))

def limiar(x):
    if x > 0:
        return 1
    else:
        return 0
