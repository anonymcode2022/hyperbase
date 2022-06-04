import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from matplotlib.lines import Line2D  
from scipy.spatial import distance
import numpy as np
from numpy.linalg import norm
from scipy.spatial import distance
import pandas as pd
import seaborn as sn

# Cosine similarity between two complex vectors a,b
def cosineSimiliraty(a,b):
    return np.dot(a, np.conj(b))/(norm(a)*norm(b))

# For display purposes
def surf_plot(dist):
    fig, ax = plt.subplots(subplot_kw = {"projection": "3d"}, figsize = (12, 8))

    X = np.arange(0, len(dist), 1)
    Y = np.arange(0, len(dist[0]), 1)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X, Y, dist, cmap = cm.coolwarm, linewidth = 0, antialiased = False)

    plt.gca().invert_yaxis()
    plt.show()
    
def level(x, q):
    lev = np.round(x * q)
    if(lev < q):
         lev += 1;
    return int(lev) - 1

def shp(a):
    print(a.shape)
    
def plot(mapp,numOfClasses,nodes,DICTIONARY):
    # Supports upto 15 classes
    markers = [x[0] for x in list(Line2D.filled_markers)]
    fig = plt.figure()
    ax = fig.add_subplot()
    x = [-1, -1, nodes, nodes]
    y = [-1, nodes, -1,nodes]
    ax.scatter(x, y, marker = '.', color = 'none')
    for i in range(10):
        ind = np.argwhere(mapp == (i + 1))
        x = [t[0] for t in ind]
        y = [t[1] for t in ind]
#         if()
        ax.scatter(x, y, label = DICTIONARY[i+1])
    plt.legend()
    plt.title("Projection")
    plt.gcf().set_size_inches(8, 8)
    plt.show()
    plt.clf()
    plt.cla()
    plt.close()
    
def R2P(x):
    return abs(x), np.angle(x)   

def build2D_map(sim_nodes,nodes, X =[] , Y= []):
     
    nodes_tot= nodes*nodes
    if(len(X)==0):
        X = 2 * np.pi * np.random.rand(1, d) # random angles
        X = 1 * np.exp(1j * X) # create a phasor vector
    if(len(Y)==0):
        Y = 2 * np.pi * np.random.rand(1, d) # random angles
        Y = 1 * np.exp(1j * Y) # create a phasor vector
    # print(X==Y)

    HDind = np.zeros((nodes_tot, d), np.cfloat)
    HDind_coord = np.zeros((nodes_tot, 2), float)
    HDindReverse={}
    cnt = 0
    for i in range(nodes):
        for j in range(nodes):
            tmp = np.power(X, (sim_nodes * (i+1))) * np.power(Y, (sim_nodes * (j+1)))
#             tmp = circular_convolution(power(X, (sim_nodes * (i+1))) , power(Y, (sim_nodes * (j+1))))
            HDind[cnt] = tmp;
            HDind_coord[cnt][0] = i
            HDind_coord[cnt][1] = j
            HDindReverse[(i,j)]=cnt
            cnt += 1
    return [HDind,HDind_coord,HDindReverse,X,Y]


def build3D_map(sim_nodes,nodes, X =[] , Y= [], Z=[]):
     
    nodes_tot= nodes*nodes*nodes
    if(len(X)==0):
        X = 2 * np.pi * np.random.rand(1, d) # random angles
        X = 1 * np.exp(1j * X) # create a phasor vector
    if(len(Y)==0):
        Y = 2 * np.pi * np.random.rand(1, d) # random angles
        Y = 1 * np.exp(1j * Y) # create a phasor vector
    if(len(Z)==0):
        Z = 2 * np.pi * np.random.rand(1, d) # random angles
        Z = 1 * np.exp(1j * Z) # create a phasor vector
    # print(X==Y)

    HDind = np.zeros((nodes_tot, d), np.cfloat)
    HDind_coord = np.zeros((nodes_tot, 3), float)
    HDindReverse={}
    cnt = 0
    for i in range(nodes):
        for j in range(nodes):
            for k in range(nodes):
                tmp = np.power(X, (sim_nodes * (i+1))) * np.power(Y, (sim_nodes * (j+1)))* np.power(Z, (sim_nodes * (k+1)))
    #             tmp = circular_convolution(power(X, (sim_nodes * (i+1))) , power(Y, (sim_nodes * (j+1))))
                HDind[cnt] = tmp;
                HDind_coord[cnt][0] = i
                HDind_coord[cnt][1] = j
                HDind_coord[cnt][2] = k
                HDindReverse[(i,j,k)]=cnt
                cnt += 1
    return [HDind,HDind_coord,HDindReverse,X,Y,Z]
def np_multiply(X, Y):
    
    output = np.empty((X.shape[0], 1), dtype=np.complex128)

    for i in range(len(X)):
        
        output[i] =np.vdot(X[i],Y)

    return output
### Original resonator implementation : binding is the HV that we want to decompose and bases are the Codebook vectors. 
def findCoordinates1(binding,bases):
    N = len(binding)
    max_iter=100
    flagPrec=0.9999;
    deg=1.01
    tot_bindings=len(bases)
    guesses=[]
    dps=[]
    flags=[]
    answer=np.zeros(tot_bindings)
    for i in range(tot_bindings):
        guess=bases[i].sum(0)
        guess/=np.abs(guess)
        guesses.append(guess)
        dps.append(np.zeros(bases[i].shape[0]))
        flags.append(0)
    for j in range(max_iter):
        for i in range(tot_bindings):
            fac_cur= binding
            for k in range(tot_bindings):
                if(i==k):
                    continue
                fac_cur=fac_cur*(np.conj(guesses[k]))
            dp_i = abs(np_multiply(bases[i],fac_cur.T).real)
            dp_i = np.power(dp_i,deg)/((np.power(dp_i,deg)).sum()); 
            guess_i_u = dp_i.T@bases[i];
            guess_i_u= guess_i_u/abs(guess_i_u);
            dps[i]=dp_i
            flag_i= np.vdot(guesses[i],guess_i_u).real/N
            flags[i] = flag_i
            guesses[i]= guess_i_u
        finished = True
        for i in range(tot_bindings):
            if(flags[i]<flagPrec):
                finished=False
        if(finished):
            break
    for i in range(tot_bindings):
        answer[i] = np.argmax(dps[i])  
    return [answer,1]
### optimized resonator implementation : binding is the HV that we want to decompose and bases are the Codebook vectors. 
def findCoordinates(binding,bases):
    numCBs = 5
    max_iter=100
    flagPrec=0.9999;
    deg=1.01
    BasesIter = []
    q= bases.shape[1]
    tot_bindings=len(bases)
    start, end = 0 , q
    toSearch =[]
    for kk in range(tot_bindings):
        toSearch.append((start, end))
    while(1):
        
        iterArray = []
        for j in range(tot_bindings):
            baseArray = []
            start,end = toSearch[j]
            for i in range(start,end,(end-start)//numCBs):
                tem = bases[j][i:i+(end-start)//numCBs].sum(0)
                baseArray.append(tem)
            iterArray.append(baseArray)    
        answer,work=findCoordinates1(binding,np.array(iterArray))
        for j in range(tot_bindings):
            answer_J  = int(answer[j])
            start,end = toSearch[j]
            block = (end-start)//numCBs
            toSearch[j] = (start+answer_J*block,start+(answer_J+1)*block)
        if(block<=q or block%q!=0):
            finalIter = []
            for j in range(tot_bindings):
                start,end = toSearch[j]
                finalIter.append(bases[j][start:end])
            break
    answer,work=findCoordinates1(binding,np.array(finalIter))
    
    factor = np.ones(bases.shape[2]).astype('complex128')
    finalAnswer = []
    for j in range(tot_bindings):
        finalAnswer.append(toSearch[j][0]+answer[j])
        factor*= bases[j][int(finalAnswer[-1])]
    work=False
    if(cosineSimiliraty(factor,binding).real>=0.5):
        work =True
    return [np.array(finalAnswer),work]