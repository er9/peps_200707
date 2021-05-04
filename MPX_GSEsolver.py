import sys
from io import StringIO
import numpy as np
from scipy import linalg
import time

import tens_fcts as tf
import MPX


# maxIt = 100


def targetState(left,right,H,psi_shape,target,isH=True):
    '''
    left:      left block
    right:     right block
    H:         MPO(ndarray) in between left and right block
    psi_shape: expected shape of psi
    '''   

    m = psi_shape[0]
    n = psi_shape[-1]

    sqdim = int(np.prod(psi_shape))

    temp = np.einsum('Lal,audb-> Ludbl',left,H)
    sys_mat = np.einsum('Ludbl,Rbr->LuRldr',temp,right).reshape(sqdim,sqdim)
    # sys_mat = np.einsum('Lal,audb,Rbr->LulRdr',left,H,right).reshape(sqdim,sqdim)

    target = target%sqdim
    if isH:
        eVals, eVecs = linalg.eigh(sys_mat,eigvals=(target,target))
        # print eVals
        return eVals[0], eVecs[:,0].reshape(psi_shape)
    else:
        eVals, eVecs = linalg.eig(sys_mat)
        k = np.argsort( np.abs(eVals)*np.sign(eVals) )[target]
        eVal_k = eVals[k]
        eVec_k = eVecs[:,k].reshape(psi_shape)
        return eVal_k, eVec_k


## ordered contraction
def addSiteToL(left,psi,op):
    temp = np.einsum('Lal,audb->Ludbl', left, op)
    temp = np.einsum('Ludbl,LuR->Rdbl', temp, np.conj(psi))
    temp = np.einsum('Rdbl,ldr->Rbr', temp, psi)
    return temp
    
def addSiteToR(psi,op,right):
    temp = np.einsum('audb,Rbr->Raudr', op, right)
    temp = np.einsum('LuR,Raudr->Ladr', np.conj(psi), temp)
    temp = np.einsum('ldr,Ladr->Lal', psi, temp)
    return temp


## main run method
def solver(H, target=0, DMAX=100, conv_err=1.0e-8,isH=True,maxIt=100):
    # H:  MPO to find eval, evec for
    # target:  which eigenvalue to target (0: minimum, -1: maximum)

    mpstate = MPX.rand([d[0] for d in H.phys_bonds],DMAX)
    mpstate = MPX.normalize(mpstate)
    mpstate = MPX.compress(mpstate,-1,1)[0]

    H2 = H.dot(H)
    L = len(mpstate)
    
    # list of length L+1 with values of blocks of different lengths
    rBlocks = [np.ones((1,1,1))]
    lBlocks = [np.ones((1,1,1))]

    rHHBlocks = [np.ones((1,1,1))]
    lHHBlocks = [np.ones((1,1,1))]
    
    ## build rBlocks with initial mpstate
    for i in range(L)[::-1]:
        rBlocks.append( addSiteToR(mpstate[i],H[i],rBlocks[L-1-i]) )
        # rBlocks.append( np.einsum('LuR,audb,ldr,Rbr->Lal', np.conj(mpstate[i]),H[i],mpstate[i],rBlocks[L-1-i]) )
    rBlocks = rBlocks[::-1]

    ## build rHHBlocks with initial mpstate
    for i in range(L)[::-1]:
        rHHBlocks.append( addSiteToR(mpstate[i],H2[i],rHHBlocks[L-1-i]) )
        # rHHBlocks.append( np.einsum('LuR,audb,ldr,Rbr->Lal', np.conj(mpstate[i]),H2[i],mpstate[i],rHHBlocks[L-1-i]) )
    rHHBlocks = rHHBlocks[::-1]
 

    it = 0
    expH = 100
    expHH = 0
    while not (np.abs(np.abs(expH)-np.sqrt(expHH))<conv_err) and it < maxIt:

        it += 1
        print [s.shape for s in mpstate]

        ### right sweep ###
        for i in range(1,L+1):

            RB = rBlocks[i]
            LB = lBlocks[i-1]

            psi_shape = [LB.shape[0],H.phys_bonds[i-1][0],RB.shape[0]]
            # psi_shape = mpstate[i-1].shape
            eVal, psi = targetState(LB,RB,H[i-1],psi_shape,target,isH)
            mpstate[i-1] = psi.copy()
            mpstate[i-1:i+1] = MPX.compress(mpstate[i-1:i+1],-1,0)[0]
            
            try:
                # lBlocks[i] = addSiteToL(LB,psi,H[i-1])
                lBlocks[i] = addSiteToL(LB,mpstate[i-1],H[i-1])
            except(IndexError):
                # lBlocks.append( addSiteToL(LB,psi,H[i-1]) )
                lBlocks.append( addSiteToL(LB,mpstate[i-1],H[i-1]) )

            try:
                # lHHBlocks[i] = addSiteToL(lHHBlocks[i-1], psi, H2[i-1])
                lHHBlocks[i] = addSiteToL(lHHBlocks[i-1], mpstate[i-1], H2[i-1])
            except(IndexError):
                # lHHBlocks.append( addSiteToL(lHHBlocks[i-1], psi, H2[i-1]) )
                lHHBlocks.append( addSiteToL(lHHBlocks[i-1], mpstate[i-1], H2[i-1]) )


        ### left sweep ###
        for i in range(L-1,-1,-1):
           RB = rBlocks[i+1]
           LB = lBlocks[i]

           psi_shape = [LB.shape[0],H.phys_bonds[i][0],RB.shape[0]]
           eVal, psi = targetState(LB,RB,H[i],psi_shape,target,isH)
           mpstate[i] = psi.copy()
           mpstate[i-1:i+1] = MPX.compress(mpstate[i-1:i+1],-1,1)[0]
           
           ##  rBlocks[i]   = np.einsum('LuR,audb,ldr,Rbr->Lal', np.conj(psi), H[i-1],  psi, RB)
           ##  rHHBlocks[i] = np.einsum('LuR,audb,ldr,Rbr->Lal', np.conj(psi), H2[i-1], psi, rHHBlocks[i-1]) 
           # rBlocks[i]   = addSiteToR( psi, H[i],  RB)
           # rHHBlocks[i] = addSiteToR( psi, H2[i], rHHBlocks[i+1]) 
           rBlocks[i]   = addSiteToR( mpstate[i], H[i],  RB)
           rHHBlocks[i] = addSiteToR( mpstate[i], H2[i], rHHBlocks[i+1]) 


        expH  = np.asscalar(rBlocks[0])
        expHH = np.asscalar(rHHBlocks[0])

        print ('iter: ', it, eVal, np.sqrt(expHH), np.abs(expH)-np.sqrt(expHH))

    if it < maxIt:     print 'successfully converged'
    else:                  print 'reached max iteration'

    print('targeted state energy: ', eVal)
    print('state shape: ', [s.shape[0] for s in mpstate])
    
    return mpstate, eVal
