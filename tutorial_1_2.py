import numpy as np
import matplotlib.pyplot as plt

# 2a)

def f(x):
    return np.exp(-x) / (1+ (x-1)**2)

b= 10
N = 10000
x = np.random.rand(N)*b

f_avg = np.mean(f(x))
I = f_avg * b

print('--------------- 2a ----------------')
print('I: ', I)
print('std of I: ', np.std(f(x)*b))

# 2b)


alpha = 1.46


def g(x):
    return alpha * np.exp(-alpha*x)

u = np.random.rand(N)
G_inverse = -1/alpha * np.log(1-u)

I = sum(f(G_inverse) / g(G_inverse)) / N

print('--------------- 2b ----------------')
print('I: ', I)
print('std of I: ', np.std(f(G_inverse) / g(G_inverse)))
