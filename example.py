"""
See how to use SPEAII module in this example.
"""

import numpy as np
import matplotlib.pyplot as plt
from SPEAII import spea2, scale, tradeoff

n=2             # number of design variables
l=n*16          # length of the chromosomes
g=100           # number of generations              
p=35            # population size
ap=35           # archive population size
bo1=0           # lower boundary of all variables
bo2=1           # upper boundary of all variables
mp=0.2          # permutation probability (0<=mp<=1)
mode='min'      # minimzation mode

from math import pi
def test(x):
    f=np.zeros((x.shape[0],2))
    f[:,0]=x[:,0]
    f[:,1]=(1+10*x[:,1])*(1-(x[:,0]/(1+10*x[:,1]))**2-(x[:,0]/(1+10*x[:,1]))*np.sin(2*pi*4*x[:,0]))    
    return f

apop,avs,afvs = spea2(test,n,l,bo1,bo2,g,p,ap,mp,prnt=1,savedata=0)

plt.plot(afvs[:,0],afvs[:,1],'bo',markersize=2,alpha=1,label='data')
plt.title('Pareto frontier after 100 generations')
plt.xlabel('f_0')
plt.ylabel('f_1')
