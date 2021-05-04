import numpy as np
import time
import tens_fcts as tf

import MPX
import PEPX


########################################################
####               get boundary MPOs                ####

''' get boundaries by contracting rows/cols of pepx 
    obtain list of boundary MPOs/MPSs,
    up to row/col "upto"
    returns err from compression of MPO/MPS if desired
'''
########################################################


def get_next_boundary_O(pep0_row,boundary,XMAX=100):
    '''  get outside boundary (facing in)
         note:  pep0_row is 1-D, has no dangling physical bonds (ie traced out)
    '''
    
    L2 = len(pep0_row)
    boundary_mpo = boundary.copy()

    for j in range(L2):
        # tens  = np.einsum('lioruu->lior',pepx_row[j])
        tens2 = np.einsum('lior,LoR->lLirR',pep0_row[j],boundary_mpo[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')
    
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,0)
    
    return boundary_mpo, err
     

def get_next_boundary_I(pep0_row,boundary,XMAX=100):
    ''' get inside boundary (facing out)
        note:  pepx_next is 1-D
    '''

    L2 = len(pep0_row)
    boundary_mpo = boundary.copy()

    for j in range(L2):
        # tens  = np.einsum('lioruu->lior',pepx_row[j])
        tens2 = np.einsum('lior,LiR->lLorR',pep0_row[j],boundary_mpo[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')
    
    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    return boundary_mpo, err


def get_next_boundary_L(pep0_col,boundary,XMAX=100):
    ''' get left boundary (facing right)
        note:  pepx_next is 1-D
    '''

    L1 = len(pep0_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):
        # tens  = np.einsum('lioruu->ilro',pepx_col[i])
        tens2 = np.einsum('IlO,lior->IirOo',boundary_mpo[i],pep0_col[i])
        boundary_mpo[i] = tf.reshape(tens2,'ii,i,ii')
    
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,0)

    return boundary_mpo, err


def get_next_boundary_R(pep0_col,boundary,XMAX=100):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''

    L1 = len(pep0_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):
        # tens  = np.einsum('lioruu->ilro',pepx_col[i])
        # tens2 = np.einsum('lior,IrO->iIloO',pep0_col[i],boundary_mpo[i])
        tens2 = np.einsum('lior,IrO->IilOo',pep0_col[i],boundary_mpo[i])
        boundary_mpo[i] = tf.reshape(tens2,'ii,i,ii')
    
    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    return boundary_mpo, err


#####################
#### full method ####
#####################

def get_boundaries(pep0,side,upto,XMAX=100,get_err=False):
    ''' get_boundaries for pep0 (no dangling physical bonds, e.g. traced out) 
    '''

    L1, L2 = pep0.shape

    if side in ['o','O',2]:

        envs = [ MPX.ones([(1,)]*L2) ]     # initialize with empty boundary
        errs = [ 0. ]

        for i in range(L1-1,upto-1,-1):      # building envs from outside to inside
            boundary_mpo, err = get_next_boundary_O(pep0[i,:], envs[-1], XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]
            
    elif side in ['i','I',1]:

        envs = [ MPX.ones([(1,)]*L2) ]     # initialize with empty boundary
        errs = [ 0. ]
     
        for i in range(upto):              # building envs from inside to outside
            boundary_mpo, err = get_next_boundary_I(pep0[i,:], envs[-1], XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
    
    elif side in ['l','L',0]:

        envs = [ MPX.ones([(1,)]*L1) ]     # initialize with empty boundary
        errs = [ 0. ]
    
        for j in range(upto):              # building envs from left to right
            boundary_mpo, err = get_next_boundary_L(pep0[:,j],envs[-1], XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
       
    elif side in ['r','R',3]:

        envs = [ MPX.ones([(1,)]*L1) ]     # initialize with empty boundary]
        errs = [ 0. ]
    
        for j in range(L2-1,upto-1,-1):      # building envs from right to left
            boundary_mpo, err = get_next_boundary_R(pep0[:,j],envs[-1], XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]
    
    else:
        print 'get_boundaries:  provide valid direction:  i,o,l,r'
        exit()
    
    if get_err:   return envs, errs
    else:         return envs




############################################################################
############################################################################
#############            get subboundary                     ###############

''' given boundaries, obtain "sub-boundaries" by filling in the rest of the pepx tens
    eg. if boundaries are rows, then get boundaries by connecting the two via the columns 
    up to row/col "upto"

    pepx_sub:  relevant subsection of pepx (between bound1 and bound2)
    returns e_mpo (an mpo/mps)
'''
#############################################################################
#############################################################################

def get_next_subboundary_O(env_mpo,boundL_tens,pep0_row,boundR_tens,XMAX=100):

    L2 = len(pep0_row)
    e_mpo = env_mpo.copy()

    for j in range(L2):
        # tens = np.einsum('lioruu,LoR->lLirR', pepx_row[j],e_mpo[j])
        tens = np.einsum('lior,LoR->lLirR', pep0_row[j],e_mpo[j])
        e_mpo[j] = tens    # tf.reshape(tens,'ii,i,ii')

    # # contract envs with boundary 1, boundary 2
    # e_mpo[0]  = np.einsum('al,lur->aur',tf.reshape(boundL_tens,'i,ii'),e_mpo[0])
    # e_mpo[-1] = np.einsum('lur,br->lub',e_mpo[-1],tf.reshape(boundR_tens,'ii,i'))    

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wlL,lLirR->wirR',boundL_tens,e_mpo[0])   # i(rRo) -- (lLx)oO(rRy)
    e_mpo[-1] = np.einsum('...rR,zrR->...z',e_mpo[-1],boundR_tens)      # (lLx)oO(rRy) -- i(lLo)

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,ii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,1)
    e_mpo, err = MPX.compress(e_mpo,XMAX,0)

    return e_mpo, err


def get_next_subboundary_I(env_mpo,boundL_tens,pep0_row,boundR_tens,XMAX=100):

    L2 = len(pep0_row)
    e_mpo = env_mpo.copy()
    
    for j in range(L2):
        # tens = np.einsum('LiR,lioruu->LloRr', e_mpo[j],pepx_row[i,j])
        tens = np.einsum('LiR,lior->LloRr', e_mpo[j],pep0_row[j])
        e_mpo[j] = tens   # tf.reshape(tens,'ii,i,ii')

    # # contract envs with boundary 1, boundary 2
    # e_mpo[0]  = np.einsum('la,lur->aur',tf.reshape(boundL_tens,'ii,i'),e_mpo[0])
    # e_mpo[-1] = np.einsum('lur,rb->lub',e_mpo[-1],tf.reshape(boundR_tens,'ii,i'))
    
    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('Llw,LloRr->woRr',boundL_tens,e_mpo[0])   # (irR)o -- (xlL)iI(yrR)
    e_mpo[-1] = np.einsum('...Rr,Rrz->...z',e_mpo[-1],boundR_tens)      # (xlL)iI(yrR) -- (ilL)o

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,ii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress(e_mpo,XMAX,1)

    return e_mpo, err


def get_next_subboundary_L(env_mpo,boundI_tens,pep0_col,boundO_tens,XMAX=100):

    L1 = len(pep0_col)
    e_mpo = env_mpo.copy()

    # print 'subL'
    # print [m.shape for m in e_mpo]
    # print L1, [m.shape for m in pep0_col]

    
    for i in range(L1):
        # tens = np.einsum('IlO,lioruu->IirOo', e_mpo[i],pepx_col[i,j])
        # print 'subbound L', e_mpo[i].shape, pep0_col[i].shape
        tens = np.einsum('IlO,lior->IirOo', e_mpo[i],pep0_col[i])
        e_mpo[i] = tens  # tf.reshape(tens,'ii,i,ii')

    # # contract envs with boundary 1, boundary 2
    # e_mpo[0]  = np.einsum('ia,iro->aro',tf.reshape(boundI_tens,'ii,i'),e_mpo[0])
    # e_mpo[-1] = np.einsum('iro,ob->irb',e_mpo[-1],tf.reshape(boundO_tens,'ii,i'))

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('Iiw,IirOo->wrOo',boundI_tens,e_mpo[0])    # l(oOr) -- (xiI)rR(yoO)
    e_mpo[-1] = np.einsum('...Oo,Ooz->...z',e_mpo[-1],boundO_tens)       # (xiI)rR(yoO) -- (liI)r

    if L1 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,ii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'ii,i,ii')
    
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress(e_mpo,XMAX,1)

    return e_mpo, err


def get_next_subboundary_R(env_mpo,boundI_tens,pep0_col,boundO_tens,XMAX=100):

    L1 = len(pep0_col)
    e_mpo = env_mpo.copy()

    # print 'subR'
    # print [m.shape for m in e_mpo]
    # print L1, [m.shape for m in pep0_col]

    for i in range(L1):
        # tens = np.einsum('lioruu,IrO->IilOo', pepx_col[i,j],e_mpo[i])
        # print 'subbound R', pep0_col[i].shape, e_mpo[i].shape
        tens = np.einsum('lior,IrO->iIloO', pep0_col[i],e_mpo[i])
        e_mpo[i] = tens     # tf.reshape(tens,'ii,...,ii')

    # # contract envs with boundary 1, boundary 2
    # e_mpo[0]  = np.einsum('ai,ilo->alo',tf.reshape(boundI_tens,'i,ii'),e_mpo[0])
    # e_mpo[-1] = np.einsum('ilo,bo->ilb',e_mpo[-1],tf.reshape(boundO_tens,'i,ii'))

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wiI,iIloO->wloO',boundI_tens,e_mpo[0])    # l(oOr) -- (iIx)lL(oOy)
    e_mpo[-1] = np.einsum('...oO,zoO->...z',e_mpo[-1],boundO_tens)       # (iIx)lL(oOy) -- l(iIr)

    if L1 > 1:    # if L1 == 1 don't need to do any reshaping
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,ii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'ii,i,ii')
    
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress(e_mpo,XMAX,1)

    return e_mpo, err


#####################
#### full method ####
#####################

def get_subboundaries(bound1,bound2,pep0_sub,side,upto,XMAX=100,get_errs=False):
    
    # print 'get subboundaries', [m.shape for idx,m in np.ndenumerate(pep0_sub)]
    L1, L2 = pep0_sub.shape  

    if side in ['o','O',2]:    # bound1, bound2 are vertical mpos (left and right side)
        envs = [ MPX.ones([(1,)]*L2) ]
        errs = [ 0. ]
        for i in range(L1-1,upto-1,-1):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_O(envs[-1], bound1[i], pep0_sub[i,:], bound2[i], XMAX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]

    elif side in ['i','I',1]:   # bound1, bound2 are vertical mpos (left and right side)
        envs = [ MPX.ones([(1,)]*L2) ]
        errs = [ 0. ]
        for i in range(upto):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_I(envs[-1], bound1[i], pep0_sub[i,:], bound2[i], XMAX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['l','L',0]:   # bound1, bound2 are horizontal mpos (top and bottom)
        envs = [ MPX.ones([(1,)]*L1) ]
        errs = [ 0. ]
        for j in range(upto):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_L(envs[-1], bound1[j], pep0_sub[:,j], bound2[j], XMAX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['r','R',3]:   # bound1, bound2 are horizontal mpos (top and bottom)
        envs = [ MPX.ones([(1,)]*L1) ]
        errs = [ 0. ]
        for j in range(L2-1,upto-1,-1):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_R(envs[-1], bound1[j], pep0_sub[:,j], bound2[j], XMAX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]


    if get_errs:   return envs, errs
    else:          return envs




#############################################
#####         contract full env     #########
#############################################

def trace(pepo):
    ''' traces over all sites, but leaves uncontracted '''

    pep0 = np.empty(pepo.shape,dtype=object)
    for ind, ptens in np.ndenumerate(pepo):
        pep0[ind] = np.einsum('...ii->...',ptens)

    # return PEPX.PEPX(pep0)
    return pep0


def contract(pepx,side='I',XMAX=100,get_err=False):
    ''' contract virtual bonds of pep0 (no physical bonds) '''
   
    L1, L2 = pepx.shape

    if   side in ['i','I',1]:   upto = L1
    elif side in ['l','L',0]:   upto = L2
    else:                       upto = 0

    # if np.all( [len(dp) == 0 for dp in pepx.phys_bonds.flat] ):
    #     boundaries = get_boundaries(pepx,side,upto,XMAX,get_err)
    # elif np.all( [len(dp) == 2 for dp in pepx.phys_bonds.flat] ):
    #     tr_pep0 = trace(pepx)
    #     boundaries = get_boundaries(tr_pep0,side,upto,XMAX,get_err)
    # else:
    #     print 'in env_rho contract:  please use pepo or pep0'
    #     exit()

    # assumes that there are no physical bonds
    boundaries = get_boundaries(pepx,side,upto,XMAX,get_err)
    
    if   side in ['i','I',1,'l','L',0]:        bound = boundaries[-1]
    else:                                      bound = boundaries[0]

    # for b in boundaries:
    #     print [t.shape for t in b]

    ntens = np.einsum('lur->lr',bound[0])
    for i in range(1,len(bound)):
        b_ = np.einsum('rUs->rs',bound[i])         # U should be dim 1
        ntens = np.einsum('lr,rs->ls',ntens,b_)

    return np.squeeze(ntens)
 
       
def embed_sites_contract(sub_pep0, envs_list, side='L',XMAX=100,get_errs=False):
    ''' get ovlp of sub_pep0 embedded in env
        useful for measuring energy and such
    '''

    L1, L2 = sub_pep0.shape
    bL, bI, bO, bR = envs_list

    embedded_pepx = sub_pep0.copy()

    cum_err = 0
    if   side in ['l','L',0]:
        sb = bL
        for j in range(L2):
            subbound, err = get_next_subboundary_L(sb,bI[j],sub_pep0[:,j],bO[j],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bR)

    elif side in ['r','R',3]:
        sb = bR
        for j in range(L2)[::-1]:
            subbound, err = get_next_subboundary_R(sb,bI[j],sub_pep0[:,j],bO[j],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bL)

    elif side in ['i','I',1]:
        sb = bI
        for i in range(L1):
            subbound, err = get_next_subboundary_I(sb,bL[i],sub_pep0[i,:],bR[i],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bO)

    elif side in ['o','O',2]:
        sb = bO
        for i in range(L1)[::-1]:
            subbound, err = get_next_subboundary_O(sb,bL[i],sub_pep0[i,:],bR[i],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bI)

    if get_errs:    return ovlp, cum_err
    else:           return ovlp


def contract_2_bounds(bound1, bound2):
    ''' eg. bL -- bR, bI -- bO '''

    output = np.einsum('lur,LuR->lLrR',bound1[0],bound2[0])
    for m in range(1,len(bound1)):
        output = np.einsum('lLrR,rus,RuS->lLsS',output,bound1[m],bound2[m])
    
    return np.einsum('llss->',output)


####### probably don't need this stuff for DM (since just use SVD) #########


############################################################################
############################################################################
#############                 get env                        ###############

''' embeds subsystem of pepx into enviornment, given by boundary mpo/mps's
    returns env:  a ring mapped to a 1-D mpo
'''
#############################################################################
#############################################################################

def embed_site(sub_pepx, boundL, boundI, boundO, boundR):

    L1, L2 = sub_pepx.shape

    embedded_pepx = sub_pepx.copy()
 
    for i in range(1,L1-1):        # apply left and right boundaries
        tens = np.einsum('IlO,lior...->IirOo...',boundL[i],sub_pepx[i,0])
        di = sub_pepx[i,0].shape[1] * boundL[i].shape[0]
        do = sub_pepx[i,0].shape[2] * boundL[i].shape[2]
        dr = sub_pepx[i,0].shape[3]
        embedded_pepx[i,0] = tens.reshape((1,di,do,dr)+tens.shape[-2:])

        tens = np.einsum('lior...,IrO->iIloO...',sub_pepx[i,-1],boundR[i])
        dl = sub_pepx[i,-1].shape[0]
        di = sub_pepx[i,-1].shape[1] * boundR[i].shape[0] 
        do = sub_pepx[i,-1].shape[2] * boundR[i].shape[2] 
        embedded_pepx[i,-1] = tens.reshape((d1,di,do,1)+tens.shape[-2:])

    for j in range(1,L2-1):        # apply inner and outer boundaries        
        tens = np.einsum('LiR,lior...->LloRr...',boundI[i],sub_pepx[0,j])
        dl = sub_pepx[0,j].shape[0] * boundI[j].shape[0]
        do = sub_pepx[0,j].shape[2]
        dr = sub_pepx[0,j].shape[3] * boundI[j].shape[2]
        embedded_pepx[0,j] = tens.reshape((dl,1,do,dr)+tens.shape[-2:])

        tens = np.einsum('lior...,LoR->lLirR...',sub_pepx[-1,j],boundO[i])
        dl = sub_pepx[-1,j].shape[0] * boundI[j].shape[0]
        di = sub_pepx[-1,j].shape[1]
        dr = sub_pepx[-1,j].shape[3] * boundI[j].shape[2]
        embedded_pepx[-1,j] = tens.reshape((dl,di,1,dr)+tens.shape[-2:])

    # corner cases
    if L1 == 1 and L2 == 1:
        tens = np.einsum('XlB,lior...,XiA->BoAr...',boundL[0],sub_pepx[0,0],boundI[0])
        tens = np.einsum('BoAr...,BoC->ArC...',tens,boundO[0])
        tens = np.einsum('ArC...,CrA->...',tens,boundR[0])
        embedded_pepx[0,0] = tens.reshape((1,1,1,1,)+tens.shape[-2:])

    elif L1 == 1 and L2 > 1:
        tens = np.einsum('XlB,lior...,XiC->BoCr...',boundL[0],sub_pepx[0,0],boundI[0])
        tens = np.einsum('BoCr...,BoA->CrA',tens,boundO[0])
        dr = sub_pepx[0,0].shape[3] * boundI[0].shape[2] * boundO[0].shape[2]
        embedded_pepx[0,0] = tens.reshape((1,1,1,dr)+tens.shape[-2:])

        tens = np.einsum('AiX,lior...,XrB->AloB...',boundI[-1],sub_pepx[-1,-1],boundR[-1])
        tens = np.einsum('AloB...,CoB->AlC...',tens,boundO[-1])
        dl = sub_pepx[-1,-1].shape[0] * boundO[-1].shape[0] * boundI[-1].shape[0]
        embedded_pepx[-1,-1] = tens.reshape((dl,1,1,1)+tens.shape[-2:])

    elif L1 > 1 and L2 == 1:
        tens = np.einsum('XlA,lior...,XiB->AoBr...',boundL[0],sub_pepx[0,0],boundI[0])
        tens = np.einsum('AoBr...,BrC->AoC',tens,boundR[0])
        do = sub_pepx[0,0].shape[2] * boundL[0].shape[2] * boundR[0].shape[2]
        embedded_pepx[0,0] = tens.reshape(1,1,do,1)

        tens = np.einsum('AlX,lior...,XoB->AirB...',boundL[-1],sub_pepx[-1,-1],boundO[-1])
        tens = np.einsum('AirB...,BrC->AiC', tens,boundR[-1])
        di = sub_pepx[-1,-1].shape[0] * boundL[-1].shape[0] * boundR[-1].shape[0]
        embedded_pepx[-1,-1] = tens.reshape((1,di,1,1)+tens.shape[-2:])

    else:
        tens = np.einsum('XlB,lior...,XiC->BoCr...',boundL[0],sub_pepx[0,0],boundI[0])
        do = sub_pepx[9,0].shape[2] * boundL[0].shape[2]
        dr = sub_pepx[0,0].shape[3] * boundI[0].shape[2]
        embedded_pepx[0,0] = tens.reshape(1,1,do,dr)
    
        tens = np.einsum('BiX,lior...,XrC->BloC...',boundI[-1],sub_pepx[0,-1],boundR[0])
        dl = sub_pepx[9,-1].shape[0] * boundI[-1].shape[0]
        do = sub_pepx[0,-1].shape[2] * boundR[0 ].shape[2]
        embedded_pepx[0,-1] = tens.reshape(dl,1,do,1)

        tens = np.einsum('BoX,lior...,CrX->lBiC...',boundO[-1],sub_pepx[-1,-1],boundR[-1])
        dl = sub_pepx[-1,-1].shape[0] * boundO[-1].shape[0]
        di = sub_pepx[-1,-1].shape[2] * boundR[-1].shape[0]	
        embedded_pepx[-1,-1] = tens.reshape(dl,di,1,1)

        tens = np.einsum('BlX,lior...,XoC->BirC...',boundL[-1],sub_pepx[-1,0],boundO[0])
        di = sub_pepx[-1,0].shape[1] * boundL[-1].shape[0]
        dr = sub_pepx[-1,0].shape[3] * boundO[0 ].shape[2]
        embedded_pepx[-1,0] = tens.reshape(1,di,1,dr)


    return embedded_pepx


def contract_embedding(embed_pepx):

   L1, L2 = embed_pepx.shape
   pass


def contract_embedded_site(sub_pepx, boundL, boundI, boundO, boundR):
    ''' contract boundaries + sub_pepx, which may contain Nones (took out the tensor) --> env tensor '''

    # contract all boundary tensors

    # skip if nan; o.w contract with boundary, neighboring tensors
    # start from non-nan corner first
    # then go by row/col (loop over whichevery is shorter)

    pass 
