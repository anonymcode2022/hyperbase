from methods import *

# Find the nth base, given that we have already found n-1 bases.
def findBaseNew(seed,TR_perm, K,currentBases,originalBindInd ):

    statistics= []
    for i in range(len(TR_perm)):
        hd_new = TR_perm[i];
        
        Ibmu = seed * np.conj(hd_new);
        
        [rho,ang] = R2P(Ibmu);  
        
        # check the correctness of the above
        temp = originalBindInd 
        ang_of_orig = R2P(temp)[1]
        ang_of_shift=(ang-ang_of_orig);

        statistics.append(ang_of_shift)

    final =False

    tempSTAT=[]
    for i in range(len(statistics)):

        ang_of_shift=statistics[i]
        if(len(currentBases)):
            tem = 1 * np.exp(1j * ang_of_shift) 
            val=0
            for kk in range(len(currentBases)):
                val+=(cosineSimiliraty(currentBases[kk],tem).real)**2
#                 val+=(cosineSimiliraty(currentBases[kk],tem).real)
            tempSTAT.append([val,i])
        else:
            finalAngle = ang_of_shift
            final=True
            FINALIND=i
            break
    if(final==False):
        finalAngle=statistics[sorted(tempSTAT)[0][1]]
        FINALIND = sorted(tempSTAT)[0][1]
    print("SELECTED sample: ", FINALIND)
    
    return 1 * np.exp(1j * finalAngle) 

# build the codebook vectors using given base vectors
def buildCodeBookVectors(sim_bases,q,bases,d):

    numBases = len(bases)
    if(numBases < 2):
        for i in range(2-numBases):
            X = 2 * np.pi * np.random.rand(1, d) # random angles
            X = 1 * np.exp(1j * X) # create a phasor vector
            bases.append(X[0])
        numBases = len(bases)

    finalBases=[]
    for i in range(numBases):
        x= np.repeat([bases[i]], q, 0)
        hd_map_f = np.power((x.T), (sim_bases * np.arange(1, q+1))).T
        finalBases.append(hd_map_f)
    return np.array(finalBases)
def genSim(bases):
    for i in range(len(bases)):
        for j in range(i+1,len(bases)):
            print("Similarity between base ",i,j," ",cosineSimiliraty(bases[i],bases[j]).real.round(3))


# HyperSeed_2 algorithm aka Hyperbase
def HyperSeed_2(TR_perm,TR_L_perm,TE_perm,TE_L_perm,sim_bases,levels,nodes,d):

    bases=[]
    for level in range(levels):
        CodeBooks = buildCodeBookVectors(sim_bases,nodes,bases,d)
        # sample to update
        updateSample = np.random.randint(low=0, high=len(TR_L_perm), size=(1,))[0]
        K = nodes//2
        sample = TR_perm[updateSample-1]
        
        originalBindInd = np.ones(d).astype('complex128')
        for i in range(len(CodeBooks)):
            
            originalBindInd *= CodeBooks[i][K-1]#   cb_baseX[K-1]*cb_baseY[K-1]
        delta = originalBindInd*sample
        seed = delta
        
        newBase = findBaseNew(seed,TR_perm, K,bases, originalBindInd)
        if(level<2):
            bases[level]=newBase
        else:
            bases.append(newBase)
    CodeBooks = buildCodeBookVectors(sim_bases,nodes,bases,d)
    
    return [seed,bases,CodeBooks]
        
        
        
        
    
                       