#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 01:57:08 2020
@author: mostafa

"""
import numpy as np
import matplotlib.pyplot as plt
from SPEAII import spea2

#------------------------------------------------------------------------------
n=2
l=n*16
g=100
p=30
ap=35
bo1=0
bo2=1
mp=0.2
mode='min'

from math import pi
def test(x):
    f=np.zeros((x.shape[0],2))

    f[:,0]=x[:,0]
    f[:,1]=(1+10*x[:,1])*(1-(x[:,0]/(1+10*x[:,1]))**2-(x[:,0]/(1+10*x[:,1]))*np.sin(2*pi*4*x[:,0]))

    #n=3,[-4,4]
    # f[:,0]=1-np.exp(-(x[:,0]-3**-0.5)**2-(x[:,1]-3**-0.5)**2-(x[:,2]-3**-0.5)**2)
    # f[:,1]=1-np.exp(-(x[:,1]+3**-0.5)**2-(x[:,2]+3**-0.5)**2)
    return f

apop,avs,afvs = spea2(test,n,l,bo1,bo2,g,p,ap,mp,prnt=1,savedata=0)

plt.plot(afvs[:,0],afvs[:,1],'bo',markersize=2,alpha=1,label='data')
