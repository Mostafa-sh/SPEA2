"""
See how to use spea2 module in this example.
"""
import numpy as np
import matplotlib.pyplot as plt
import spea2

# objective function
def test(x):
    f=np.zeros((x.shape[0],2))
    f[:,0]=x[:,0]
    f[:,1]=(1+10*x[:,1])*(1-(x[:,0]/(1+10*x[:,1]))**2-(x[:,0]/(1+10*x[:,1]))*np.sin(2*3.14159*4*x[:,0]))    
    return f

n=2      # number of real-valued design variables
bo1=0    # lower boundary of all variables. [0,0] also works.
bo2=1    # upper boundary of all variables. [1,1] also works.
g=70     # number of generations
p=40     # size of population         
ap=25    # size of archive population (archive consists of the elites in population)
mp=.1    # mutation probability (0<=mp<1)

avs,afvs = spea2.optimize(test,n,bo1,bo2,g,p,ap,mp,prnt_msg=0,savedata=1)

fig=plt.figure(1)
plt.scatter(afvs[:,0],afvs[:,1],s=15,alpha=0.7)
plt.title('Pareto frontier after '+str(g)+' generations')
plt.xlabel('function 1')
plt.ylabel('function 2')
fig.savefig('Pareto.png',dpi=150)