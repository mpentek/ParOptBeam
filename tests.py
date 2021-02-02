import numpy as np
from scipy import linalg
import math 
import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 1)
def mul (fac):
    return 2*fac
    
x= np.linspace(1,3,3)
ax.plot(x,mul(x))
k1 = 2
k2 = 1
m = 1.3
x = np.linspace(0.2, 5, 20)

def f(k1,k2,m):
    return ((2*k2 + k1) + math.sqrt(4*k2**2 +k1**2)) / (2*m)

def phi (k1,k2):
    return (2*k2 +k1 + math.sqrt(4*k2**2 + k1**2))/(2*k2) + k1/k2 +1

A = np.arange(25).reshape(5,5)
print (np.linspace(0.5,1.0,0.02))
