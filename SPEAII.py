#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 20:12:49 2019
@author: Mostafa

A simple code for genetic algorithm optimization.

Based on SPEA2: Improving the Strength Pareto
Evolutionary Algorithm proposed by E.Zitzler, M. Laumanns, and L. Thiele
https://doi.org/10.3929/ethz-a-004284029

"""

import numpy as np
import matplotlib.pyplot as plt
from os import path,mkdir

def inpop(p,l):
    # p is the population number
    # l is the length of chromosome
    pop = np.random.rand(p,l)<=0.5
    pop = pop.astype(int)
    return pop

def dcode(pop,a,b,n=1,d=None):
#     n= number of design variabels
#    "pop"=population matrix [each row must be a chromosome]
#    "n"(Optinal)=number of design variables (default: n=1)
#    "d"(Optinal)=parts of a chromosome that is allocated to design variables

    if np.isscalar(a): a=np.array([a]).astype(float)
    if np.isscalar(b): b=np.array([b]).astype(float)
    if a.size==1 and n>1: a=a*(np.zeros(n).astype(float)+1)
    if b.size==1 and n>1: b=b*(np.zeros(n).astype(float)+1)

    s=pop.shape
    if d==None:
        #dividing chromosomes equally
        d=np.zeros(n).astype(int)
        key=s[1]     #chromosomes length
        while key!=0:
            for i in np.arange(n):
                d[i]+=1
                key-=1
                if key==0:
                    break

    c=np.zeros([2,d.size]).astype(int)
    c[1]=np.cumsum(d)
    c[0,1:]=c[1,:-1]

    l=d.astype(float)        # for correct 2** calculation
    dpop=np.zeros([n,s[0]])
    for i in np.arange(n):
        v = np.sum(pop[:,c[0,i]:c[1,i]]*(2**np.flip(np.arange(l[i]),axis=0)),1)
        dpop[i] = v*(b[i]-a[i])/(2**l[i]-1)+a[i]
    dpop=dpop.T

    return dpop

def fitness(V,mode='min'):
    V=(V-np.min(V,0))/(np.max(V,0)-np.min(V,0))
    n=V.shape[0]
    T=np.swapaxes(np.tile(V,(n,1,1)),0,1)
    nE=np.logical_not( np.all(V==T,2) )
    if mode=='min':
        S=np.sum( np.all(V>=T,2) & nE, 1 )
        M=np.all(V<=T,2) & nE
    elif mode=='max':
        S=np.sum( np.all(V<=T,2) & nE, 1 )
        M=np.all(V>=T,2) & nE
    F=np.array( [np.sum(S[M[i,:]]) for i in np.arange(n)] )

    c=np.round(n**(0.5)).astype(int)
    L=np.sort(np.sum((V-T)**2,2)**0.5,1)
    l=L[:,c]
    # finding identicals and removing their distance
    I=np.all(L==np.swapaxes(np.tile(L,(n,1,1)),0,1),2)
    A=np.arange(n)
    for i in A:
        if np.sum(I[i,:])>1:
            a=A[I[i,:]]
            l[a[1:]]=-1
    D=1/(l+2)

    return F+D

def envsel(e,F,V):
    argPF=np.arange(len(F))
    argPF=argPF[F<1]
    g=argPF.shape[0]
    msg='No change in the Pareto frontier'
    if e>g:
        msg=str(e-g)+' points added to the Pareto frontier'
        argPF=np.argsort(F)[:e]

    elif e<g:
        msg=str(g-e)+' points removed from Pareto frontier'
        V=(V-np.min(V,0))/(np.max(V,0)-np.min(V,0))
        V=V[argPF]
        n=V.shape[0]
        T=np.swapaxes(np.tile(V,(n,1,1)),0,1)
        L0=np.sum((V-T)**2,2)**0.5
        L0=np.round(L0*1e15)/1e15
        for j in np.arange(g-e):
            L=np.sort(L0,1)
            c=np.arange(L.shape[0])
            L[:,0]=c
            for i in c[1:]:
                L = L[ L[:,i]==np.min(L[:,i]), :]
                if L.shape[0]==1:
                    break
            d=L[0,0].astype(int)
            L0=np.delete(L0, d, 0)
            L0=np.delete(L0, d, 1)
            argPF=np.delete(argPF,d,0)

    return argPF, msg

def newgen(pop,fit,mp,n=None):
    s=pop.shape
    p=s[0]
    l=s[1]
    if n==None: n=p

    # Binary tournament selection
    g1=np.random.randint(0,p,n)
    g2=np.random.randint(0,p,n)
    m=fit[g1]<fit[g2]
    w = np.concatenate([g1[m],g2[np.logical_not(m)]]) #winners!

    npop=pop[w,:]  #mating pool

    for i in np.arange(int(n/2)):
        part=np.arange(np.random.randint(1,l))
        c1=2*i
        c2=2*i+1

        # c1 crossover
        keep=npop[c1,part]
        npop[c1,part]=npop[c2,part]
        # c1 mutation
        if np.random.rand()<mp:
            d=np.random.randint(l)
            npop[c1,d]=1*(not npop[c1,d])

        # c2 crossover
        npop[c2,part]=keep
        # c2 mutation
        if np.random.rand()<mp:
            d=np.random.randint(l)
            npop[c2,d]=1*(not npop[c2,d])

    return npop

def cat(A,B):
    if A.shape[0]==0 and B.shape[0]==0:
        return A
    elif A.shape[0]==0:
        return B
    elif B.shape[0]==0:
        return A
    elif (A.ndim==1 and B.ndim==1) or (A.ndim==2 and B.ndim==2 and A.shape[1]==B.shape[1]):
        return np.concatenate((A,B))
    elif A.ndim==1 and B.ndim==2 and A.shape[0]==B.shape[1]:
        return np.concatenate((np.array([A]),B),axis=0)
    elif A.ndim==2 and B.ndim==1 and A.shape[1]==B.shape[0]:
        return np.concatenate((A,np.array([B])),axis=0)
    else:
        print('\nError: cat() inputs sizes are incompatible with each other!\n')

def a2s(A,sp=' '):
    if type(A)==list or type(A)==np.ndarray:
        s=str(A[0])
        for num in A[1:]:
            s+=sp+str(num)
    else:
        s=str(A)
    return s

def tradeoff(F):
    d=np.max(F,axis=0)-np.min(F,axis=0)
    d[d==0]=1
    I=np.argmin(np.linalg.norm((F-np.min(F,axis=0))/d,axis=1))
    return F[I,:],I

def spea2(objfun,n,l,bo1,bo2,g,p,ap,mp=0.2,mode='min',ipop=None,
          prnt=1,savedata=0,outfile='outputs'):

    if savedata!=0:
        out=outfile+'/'
        if path.isdir(out)==0:
            mkdir(out)
        #---------
        logfile=out+'log.txt'
        fl=open(logfile,'w+')
        fl.write("\nMOD"+' '+mode+'\n')
        fl.write("NG"+' '+str(g)+'\n')
        fl.write("NP"+' '+str(p)+'\n')
        fl.write("NA"+' '+str(ap)+'\n')
        fl.write("MP"+' '+str(mp)+'\n')
        fl.write("ND"+' '+str(n)+'\n')
        fl.write("CL"+' '+str(l)+'\n')
        fl.write("BO1"+' '+a2s(bo1)+'\n')
        fl.write("BO2"+' '+a2s(bo2)+'\n')
        fl.close()
        #---------
    if savedata==2:
        paretofile=out+'pareto.txt'
        fl=open(paretofile,'w')
        fl.write('#Indices of the Pareto Frontiers\n')
        fl.close()
    #--------------------------------
    if np.all(ipop==None):
        npop=inpop(p,l)
    else:
        npop=ipop
    nvs=dcode(npop,bo1,bo2,n)
    apop=np.array([])
    avs=np.array([])
    afvs=np.array([])
    for t in np.arange(g):
        nfvs=objfun(nvs)
        fvs=cat(afvs,nfvs)
        fit=fitness(fvs,mode)
        pf,msg=envsel(ap,fit,fvs)
        afvs=fvs[pf,:]
        vs=cat(avs,nvs)
        avs=vs[pf,:]
        pop=cat(apop,npop)
        apop=pop[pf,:]
        afit=fit[pf]
        #--------------------------------
        gn=str(t)
        if prnt==1:
            print('Generation'+' '+gn+': '+msg)
        if savedata!=0:
            fl=open(logfile,'a')
            fl.write('\nGEN'+' '+gn+'\n')
            fl.write('MSG'+' '+msg+'\n')
            tfv,ind=tradeoff(afvs)
            tv=avs[ind,:]
            fl.write('TF'+' '+a2s(tfv)+'\n')
            fl.write('TV'+' '+a2s(tv)+'\n')
            fl.close()
            name=out+'gen'+gn+'.txt'
            hdr='gen'+gn+'{'
            ftr='}gen'+gn
        if savedata==1:
            np.savetxt(name,np.concatenate((avs,afvs),axis=1),header=hdr,footer=ftr)
        elif savedata==2:
            np.savetxt(name,np.concatenate((vs,fvs),axis=1),  header=hdr,footer=ftr)
            if len(pf)<ap:
               pf1=np.array([pf]).T
               pf1=cat(pf1,np.zeros((ap-len(pf),1),dtype=int)).T
            else:
               pf1=np.array([pf])
            fl=open(paretofile,'a')
            np.savetxt(fl,pf1,fmt='%i')
            fl.close()
        #--------------------------------
        if t==g-1: break
        npop=newgen(apop,afit,mp,p)
        nvs=dcode(npop,bo1,bo2,n)
    return apop,avs,afvs

def show2f(g,m=[-2,-1],outdir='outputs'):
    if not isinstance(g, (list, tuple, np.ndarray)):
        g=np.arange(g)
    for t in g:
        f=np.loadtxt(outdir+'/gen'+str(t)+'.txt')
        plt.plot(f[:,m[0]],f[:,m[1]],'bo',markersize=2,alpha=0.5,label='data')
        plt.title('gen: '+str(t))
        plt.show()
