# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 21:26:52 2019

@author: Mostafa
"""
import numpy as np
import matplotlib.pyplot as plt
from SPEAII import inpop, Dcode, cat, fitness, envsel, newgen

def ilust(val,pf):
    plt.plot(val[:,0],val[:,1],'bo',markersize=2,alpha=1,label='data')
    plt.plot(val[pf,0],val[pf,1],'ro',markersize=2,alpha=1,label='data')
    plt.show()

#------------------------------------------------------------------------------   
n=2
l=n*16
g=100
p=60
ap=80
bo1=0
bo2=1
mp=0.3
mode='min'    
    
from math import pi 
def test(x):
    f=x
    f[:,1]=(1+10*x[:,1])*(1-(x[:,0]/(1+10*x[:,1]))**2-(x[:,0]/(1+10*x[:,1]))*np.sin(2*pi*4*x[:,0]))
    return f

npop=inpop(p,l)
apop=np.array([])
avs=np.array([])
for t in np.arange(g):
    nvs=test( Dcode(npop,bo1,bo2,n) )
    vs=cat(avs,nvs)
    fit=fitness(vs,mode)
    pf,msg=envsel(ap,fit,vs)
    pop=cat(apop,npop)
    apop=pop[pf,:]
    avs=vs[pf,:]
    afit=fit[pf]
    npop=newgen(apop,afit,mp,p)
#    print('Generation '+str(t))
#    ilust(vs,pf)
#------------------------------------------------------------------------------
ilust(avs,np.arange(len(avs)))
