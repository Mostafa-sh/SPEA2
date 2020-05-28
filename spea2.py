"""
Created on Mon Dec 16 20:12:49 2019
@author: https://github.com/mostafa-sh

A simple code for genetic algorithm optimization.

Based on SPEA2: Improving the Strength Pareto Evolutionary Algorithm 
proposed by E.Zitzler, M. Laumanns, and L. Thiele
paper: https://doi.org/10.3929/ethz-a-004284029

"""
import numpy as np
# saving data
from os import path, mkdir
from datetime import datetime

def scale(x):
    s=np.std(x,0)
    s[s==0]=1 #columns of identical elements becomes columns of 0                
    return (x-np.mean(x,0))/s
    
def fitness(V,mode='min'):
    V=scale(V)
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
    msg='No change in archive size'
    if e>g:
        msg=str(e-g)+' points added to archive'
        argPF=np.argsort(F)[:e]
    elif e<g:
        msg=str(g-e)+' points removed from archive'
        V=scale(V)
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

def newgen(var,bo1,bo2,fit,cp=0.5,eta=3,mp=0.1,sigma2=0.2,k=None):
    """
    This includes both crossover and mutation operations.
    cp is the crossover probability of each variable
    2<eta<5 commonly, larger values makes offspring variables closer to their parents
    mp is the mutation probability of each chromosome
    0<=sigma2, smaller values makes mutating variables closer to their parents
    k is the size of the new population (the new set of variables)
    """ 
    s=var.shape
    p=s[0]
    n=s[1]
    if k==None: k=p
    if k%2 !=0: o=1  # is odd
    else:       o=0
    k+=o
    #boundaries
    if np.isscalar(bo1) or len(bo1)==1: bo1=bo1*np.ones(n)
    if np.isscalar(bo2) or len(bo2)==1: bo2=bo2*np.ones(n)
    
    # Binary tournament selection
    g1=np.random.randint(0,p,k)
    g2=np.random.randint(0,p,k)
    m=fit[g1]<fit[g2]
    w=np.concatenate([g1[m],g2[np.logical_not(m)]]) #winners!
    
    #Mating pool
    nvar=var[w,:] 
    
    #Parents
    h=int(k/2)
    p1=nvar[0:h  ,:]
    p2=nvar[h:2*h,:]
    
    #Simulated Binary Crossover (SBX)
    u=np.random.rand(h,n)
    i=u<=0.5
    j=np.logical_not(i)
    
    if n==1: cp=1
    x=np.random.rand(h,n)<cp  #participating variables in parents       
    i=np.logical_and(i,x)
    j=np.logical_and(j,x)
    
    beta=np.ones((h,n))
    beta[i]=(2*u[i])      **(1/(eta+1))
    beta[j]=(1/2/(1-u[j]))**(1/(eta+1))

    c1=0.5*((1+beta)*p1+(1-beta)*p2)
    c2=0.5*((1-beta)*p1+(1+beta)*p2)
    c=np.clip( np.vstack((c1,c2[:h-o,:])), bo1,bo2)
    
    #Mutation
    k-=o
    mc=np.random.rand(k)<mp          #mutating chromosomes
    t=int(sum(mc))
    c[mc, np.random.randint(0,n,t)] += np.random.normal(0,sigma2,t)
    c[mc,:]=np.clip(c[mc,:],bo1,bo2)
    
    return c

def a2s(a,sp=' '):
    if type(a)==list or type(a)==np.ndarray:
        s=str(a[0])
        for num in a[1:]: 
            s+=sp+str(num)
    else:
        s=str(a)
    return s

def tradeoff(f):
    sf=scale(f)
    I=np.argmin(np.linalg.norm(sf,axis=1))
    return f[I,:],I

def optimize(objfun,n,bo1,bo2,
             g=60,p=20,ap=20,mp=0.1,cp=0.5,eta=3,sigma2=0.2,mode='min',ivs=None,
             prnt_msg=1,savedata=0,outfile='outputs'):
    #saving data---------------------
    if savedata!=0:
        out=outfile+'/'
        if path.isdir(out)==0:
            mkdir(out)
        #---------
        logfile=out+'log.txt'
        fl=open(logfile,'w+')
        fl.write('datetime '+datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")[:-3]+'\n\n')
        fl.write('Optimization mode:         '+mode+'\n')
        fl.write('Number of variables:       '+str(n)+'\n')
        fl.write('Lower boundary:            '+a2s(bo1)+'\n')
        fl.write('Upper boundary:            '+a2s(bo2)+'\n')
        fl.write('Number of generations:     '+str(g)+'\n')
        fl.write('Population size:           '+str(p)+'\n')
        fl.write('Archive population size:   '+str(ap)+'\n')
        fl.write('Crossover probability:     '+str(cp)+'\n')
        fl.write('Eta:                       '+str(eta)+'\n')
        fl.write('Mutation probability:      '+str(mp)+'\n')        
        fl.write('Sigma2:                    '+str(sigma2)+'\n')
        fl.close()
        #---------
    if savedata==2:
        paretofile=out+'pareto.txt'
        fl=open(paretofile,'w')
        fl.write('#Indices of the Pareto frontiers per generation\n')
        fl.close()
    #--------------------------------
    if np.all(ivs==None): vs=np.random.uniform(bo1,bo2,(p,n))
    fvs=objfun(vs)
    for t in np.arange(g):
        fit=fitness(fvs,mode)
        pf,msg=envsel(ap,fit,fvs)
        avs=vs[pf,:]
        afvs=fvs[pf,:]
        afit=fit[pf]
        #saving data----------------------
        gn=str(t+1)
        if prnt_msg==1: print('gen '+gn+': '+msg)
        if savedata!=0:
            fl=open(logfile,'a')
            fl.write('\nGeneration '+gn+'\n'+msg+'\n')
            tf,ind=tradeoff(afvs)
            fl.write('Tradeoff values    '+a2s(tf) +'\n')
            fl.write('Tradeoff variables '+a2s(avs[ind,:])+'\n')
            fl.close()
            name=out+'gen'+gn+'.txt'
            hdr='gen'+gn+'{'
            ftr='}gen'+gn
        if savedata==1:
            np.savetxt(name,np.concatenate((avs,afvs),axis=1),header=hdr,footer=ftr)
        elif savedata==2:
            np.savetxt(name,np.concatenate((vs,fvs),axis=1),  header=hdr,footer=ftr)
            fl=open(paretofile,'a')
            np.savetxt(fl,np.array([np.concatenate((pf,np.zeros(ap-len(pf),dtype=int)))]),fmt='%i')
            fl.close()
        #--------------------------------
        #TO-DO: improving stop criterion
        if t==g-1:                                   
            #saving data---------------------
            if savedata!=0:
                fl=open(logfile,'a')
                fl.write('\ndatetime '+datetime.now().strftime("%d/%m/%Y %H:%M:%S:%f")[:-3])
                fl.close()
            #--------------------------------
            break  
        nvs=newgen(avs,bo1,bo2,afit,cp,eta,mp,sigma2)
        fvs=np.vstack((afvs,objfun(nvs)))
        vs=np.vstack((avs,nvs))
    return avs,afvs