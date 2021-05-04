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


# @profile
def get_next_boundary_O(peps_u_row,peps_d_row,boundary,XMAX=100,fast_contract=False):
    '''  get outside boundary (facing in)
         note:  peps_x_row is 1-D,
    '''
    
    L2 = len(peps_u_row)
    boundary_mpo = boundary.copy()

    if fast_contract:

        boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
        errs = [[]]*(L2-1)

        for j in range(L2):
            tens2 = np.einsum('liord,xoOy->xliOyrd',peps_u_row[j],boundary_mpo[j])
            tens2 = np.einsum('xliOyrd,LIORd->xlLiIyrR',tens2,peps_d_row[j])
            boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

            if j == 0:   continue

            # twosite = np.einsum('ludr,rUDs->luUdDs',boundary_mpo[j-1],boundary_mpo[j])
            two_sites, err = MPX.compress(boundary_mpo[j-1:j+1],XMAX,0)
            boundary_mpo[j-1:j+1] = two_sites
            errs[j-1] = err[0]

        if np.sum(err) > 1.0:
            print 'bound O large compression error fast_contract'


    else:

        for j in range(L2):
            # print peps_u_row[j].shape, peps_d_row[j].shape, boundary_mpo[j].shape
            # tens2 = np.einsum('liord,LIORd,xoOy->xlLiIyrR',peps_u_row[j],peps_d_row[j],boundary_mpo[j])
            tens2 = np.einsum('liord,xoOy->xliOyrd',peps_u_row[j],boundary_mpo[j])
            tens2 = np.einsum('xliOyrd,LIORd->xlLiIyrR',tens2,peps_d_row[j])
            boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')
        
        boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
        boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,0)
        # boundary_mpo, err = MPX.compress(boundary_mpo,XMAX,0)

        if np.sum(err) > 1.0:
            print 'bound O large compression error', XMAX
            print [np.linalg.norm(m) for m in boundary_mpo]
            print err
            # cut_ind = boundary_mpo[L2/2].ndim-1
            # u,s,v,dwt = tf.svd(boundary_mpo.getSites(),cut_ind,-1)
            # print s, dwt
            # exit()

    return boundary_mpo, err
     

# @profile
def get_next_boundary_I(peps_u_row,peps_d_row,boundary,XMAX=100,fast_contract=False):
    ''' get inside boundary (facing out)
        note:  peps_x_row is 1-D
    '''

    L2 = len(peps_u_row)
    boundary_mpo = boundary.copy()

    if fast_contract:

        boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
        errs = [[]]*(L2-1)

        for j in range(L2):
            tens2 = np.einsum('xiIy,liord->xlIoyrd',boundary_mpo[j],peps_u_row[j])
            tens2 = np.einsum('xlIoyrd,LIORd->xlLoOyrR',tens2,peps_d_row[j])
            boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

            if j == 0:   continue

            # twosite = np.einsum('ludr,rUDs->luUdDs',boundary_mpo[j-1],boundary_mpo[j])
            two_sites, err = MPX.compress(boundary_mpo[j-1:j+1],XMAX,0)
            boundary_mpo[j-1:j+1] = two_sites
            errs[j-1] = err[0]

        # print 'env I final', [m.shape for m in boundary_mpo]

        if np.sum(err) > 1.0:
            print 'bound I large compression error fast_contract'


    else:

        for j in range(L2):
            # print boundary_mpo[j].shape, peps_u_row[j].shape, peps_d_row[j].shape
            # tens2 = np.einsum('xiIy,liord,LIORd->xlLoOyrR',boundary_mpo[j],peps_u_row[j],peps_d_row[j])
            tens2 = np.einsum('xiIy,liord->xlIoyrd',boundary_mpo[j],peps_u_row[j])
            tens2 = np.einsum('xlIoyrd,LIORd->xlLoOyrR',tens2,peps_d_row[j])
            boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')
        
        # print [m.shape for m in peps_u_row]
        # print 'next bi', [m.shape for m in boundary_mpo]

        # print 'boundary I', [m.shape for m in boundary_mpo]

        boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
        boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)
        # boundary_mpo, err = MPX.compress(boundary_mpo,XMAX,1)

        if np.sum(err) > 1.0:
            print 'bound I large compression error', XMAX
            print [np.linalg.norm(m) for m in boundary_mpo]
            print err
            # cut_ind = boundary_mpo[L2/2].ndim-1
            # u,s,v,dwt = tf.svd(boundary_mpo.getSites(),cut_ind,-1)
            # print s, dwt
            # exit()

    return boundary_mpo, err


# @profile
def get_next_boundary_L(peps_u_col,peps_d_col,boundary,XMAX=100):
    ''' get left boundary (facing right)
        note:  peps_x_col is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):
        # tens2 = np.einsum('xlLy,liord,LIORd->xiIrRyoO',boundary_mpo[i],peps_u_col[i],peps_d_col[i])
        tens2 = np.einsum('xlLy,liord->xLioryd',boundary_mpo[i],peps_u_col[i])
        tens2 = np.einsum('xLioryd,LIORd->xiIrRyoO',tens2,peps_d_col[i])
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')
    
    # print 'boundary L', [m.shape for m in boundary_mpo]

    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err  = MPX.compress_reg(boundary_mpo,XMAX,0)
    # boundary_mpo, err  = MPX.compress(boundary_mpo,XMAX,0)

    return boundary_mpo, err


def get_next_boundary_R(peps_u_col,peps_d_col,boundary,XMAX=100):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):
        # tens2 = np.einsum('liord,LIORd,xrRy->xiIlLyoO',peps_u_col[i],peps_d_col[i],boundary_mpo[i])
        tens2 = np.einsum('liord,xrRy->xlioRyd',peps_u_col[i],boundary_mpo[i])
        tens2 = np.einsum('LIORd,xlioRyd->xiIlLyoO',peps_d_col[i],tens2)
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)
    # boundary_mpo, err = MPX.compress(boundary_mpo,XMAX,1)

    return boundary_mpo, err


#####################
#### full method ####
#####################

def get_boundaries(peps_u,peps_d,side,upto,init_bound=None,XMAX=100,get_err=False):
    ''' get_boundaries for pep0 (no dangling physical bonds, e.g. traced out) 
    '''

    L1, L2 = peps_u.shape

    if side in ['o','O',2]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]

        for i in range(L1-1,upto-1,-1):      # building envs from outside to inside
            boundary_mpo, err = get_next_boundary_O(peps_u[i,:],peps_d[i,:],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]
            
    elif side in ['i','I',1]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]

     
        for i in range(upto):              # building envs from inside to outside
            # print 'bi peps',[m.shape for idx, m in np.ndenumerate(peps_u)]
            # print 'bi mpo',[m.shape for m in envs[-1]]
        
            boundary_mpo, err = get_next_boundary_I(peps_u[i,:],peps_d[i,:],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
    
    elif side in ['l','L',0]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L1) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(upto):              # building envs from left to right
            boundary_mpo, err = get_next_boundary_L(peps_u[:,j],peps_d[:,j],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
       
    elif side in ['r','R',3]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L1) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(L2-1,upto-1,-1):      # building envs from right to left
            boundary_mpo, err = get_next_boundary_R(peps_u[:,j],peps_d[:,j],envs[-1],XMAX)
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

# note:  ordering of grouped axes in subboundary is independent of ordering of axes in boundary
# because that connection (btwn sb and b) is  given by axis that is not grouped

def get_next_subboundary_O(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()

    for j in range(L2):
        # tens = np.einsum('liord,LIORd,xoOy->lLxiIrRy', peps_u_row[j],peps_d_row[j],e_mpo[j])
        tens = np.einsum('LIORd,xoOy->LxIoRyd',peps_d_row[j],e_mpo[j])
        tens = np.einsum('liord,LxIoRyd->lLxiIrRy',peps_u_row[j],tens)
        e_mpo[j] = tens  # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wlLx,lLxiIrRy->wiIrRy',boundL_tens,e_mpo[0])   # i(rRo) -- (lLx)oO(rRy)
    e_mpo[-1] = np.einsum('...rRy,zrRy->...z',e_mpo[-1],boundR_tens)      # (lLx)oO(rRy) -- i(lLo)

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')
    
    e_mpo  = MPX.canonicalize(e_mpo,1)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0)

    return e_mpo, err


def get_next_subboundary_I(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()
    
    for j in range(L2):
        # tens = np.einsum('xiIy,liord,LIORd->xlLoOyrR', e_mpo[j],peps_u_row[j],peps_d_row[j])
        tens = np.einsum('xiIy,liord->xlIoyrd',e_mpo[j],peps_u_row[j])
        tens = np.einsum('xlIoyrd,LIORd->xlLoOyrR',tens,peps_d_row[j])
        e_mpo[j] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('xlLw,xlLoOyrR->woOyrR',boundL_tens,e_mpo[0])   # (irR)o -- (xlL)iI(yrR)
    e_mpo[-1] = np.einsum('...yrR,yrRz->...z',e_mpo[-1],boundR_tens)      # (xlL)iI(yrR) -- (ilL)o

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')
    
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1)

    return e_mpo, err


def get_next_subboundary_L(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100):

    L1 = len(peps_u_col)

    e_mpo = env_mpo.copy()
    
    for i in range(L1):
        # print 'sub L', e_mpo[i].shape, peps_u_col[i].shape, peps_d_col[i].shape
        # tens = np.einsum('xlLy,liord,LIORd->xiIrRyoO', e_mpo[i],peps_u_col[i],peps_d_col[i])
        tens = np.einsum('xlLy,liord->xiLryod',e_mpo[i],peps_u_col[i])
        tens = np.einsum('xiLryod,LIORd->xiIrRyoO',tens,peps_d_col[i])
        e_mpo[i] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    # print 'subL', boundI_tens.shape, e_mpo[0].shape
    e_mpo[0]  = np.einsum('xiIw,xiIrRyoO->wrRyoO',boundI_tens,e_mpo[0])    # l(oOr) -- (xiI)rR(yoO)
    e_mpo[-1] = np.einsum('...yoO,yoOz->...z',e_mpo[-1],boundO_tens)       # (xiI)rR(yoO) -- (liI)r


    if L1 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')

    e_mpo  = MPX.canonicalize(e_mpo,1)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0)

    if np.sum(err) > 1.0:
        print 'subL large compression error', XMAX
        print [np.linalg.norm(m) for m in e_mpo]
        print [np.linalg.norm(m) for m in env_mpo], np.linalg.norm(boundI_tens), np.linalg.norm(boundO_tens)
        print [np.linalg.norm(m) for m in peps_u_col]
        print [np.linalg.norm(m) for m in peps_d_col]
        print err
        cut_ind = e_mpo[i].ndim-1
        u,s,v,dwt = tf.svd(e_mpo.getSites(),cut_ind,-1)
        print s, dwt
        # exit()

    return e_mpo, err


def get_next_subboundary_R(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100):

    L1 = len(peps_u_col)
    e_mpo = env_mpo.copy()

    for i in range(L1):
        # print 'br',i,L1,e_mpo[i].shape, peps_u_col[i].shape, peps_d_col[i].shape
        # tens = np.einsum('liord,LIORd,xrRy->iIxlLoOy', peps_u_col[i],peps_d_col[i],e_mpo[i])
        tens = np.einsum('LIORd,xrRy->IxLrOyd',peps_d_col[i],e_mpo[i])
        tens = np.einsum('liord,IxLrOyd->iIxlLoOy',peps_u_col[i],tens)
        e_mpo[i] = tens    # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wiIx,iIxlLoOy->wlLoOy',boundI_tens,e_mpo[0])    # l(oOr) -- (iIx)lL(oOy)
    e_mpo[-1] = np.einsum('...oOy,zoOy->...z',e_mpo[-1],boundO_tens)       # (iIx)lL(oOy) -- l(iIr)

    if L1 > 1:    # if L1 == 1 don't need to do any reshaping
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')
    
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1)

    return e_mpo, err




#####################
#### full method ####
#####################

def get_subboundaries(bound1,bound2,peps_u_sub,peps_d_sub,side,upto,init_sb=None,XMAX=100,get_errs=False):
    
    L1, L2 = peps_u_sub.shape  

    if side in ['o','O',2]:    # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(L1-1,upto-1,-1):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_O(envs[-1],bound1[i],peps_u_sub[i,:],peps_d_sub[i,:],bound2[i],XMAX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]

    elif side in ['i','I',1]:   # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(upto):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_I(envs[-1],bound1[i],peps_u_sub[i,:],peps_d_sub[i,:],bound2[i],XMAX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['l','L',0]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L1) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(upto):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_L(envs[-1],bound1[j],peps_u_sub[:,j],peps_d_sub[:,j],bound2[j],XMAX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['r','R',3]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L1) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(L2-1,upto-1,-1):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_R(envs[-1],bound1[j],peps_u_sub[:,j],peps_d_sub[:,j],bound2[j],XMAX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]


    if get_errs:   return envs, errs
    else:          return envs


#############################################
#####     ovlps from boundaries     #########
#############################################

def ovlp_from_bound(bound,to_squeeze=True):
    ''' obtain ovlp from bound which is from contracting entire peps by row/col
        should be mpo with physical bonds (1,1)
    '''

    ovlp = np.einsum('luur->lr',bound[0])
    for m in bound[1:]:
        ovlp = np.einsum('lR,Ruur->lr',ovlp,m)

    if to_squeeze:
        return np.squeeze(ovlp)
    else:
        return ovlp


def contract_2_bounds(bound1, bound2):
    ''' eg. bL -- bR, bI -- bO '''

    output = np.einsum('ludr,LudR->lLrR',bound1[0],bound2[0])
    for m in range(1,len(bound1)):
        # output = np.einsum('lLrR,ruds,RudS->lLsS',output,bound1[m],bound2[m])
        output = np.einsum('lLrR,ruds->lLRuds',output,bound1[m])
        output = np.einsum('lLRuds,RudS->lLsS',output,bound2[m])
    
    return np.einsum('llss->',output)



#############################################
#####         contract full env     #########
#############################################

## can also be done from PEPX class, dotting bra+ket and calling PEP0_env contract fct ##


## alternative method (following boundary method more closely ##
# @profile
def get_ovlp(pepx_u, pepx_d, side='I',XMAX=100,get_err=False):
    ''' contract virtual bonds of pep0 (no physical bonds) '''
   
    L1, L2 = pepx_u.shape

    # print 'ovlp', [m.shape for idx,m in np.ndenumerate(pepx_u)]

    if   side in ['i','I',1]:   upto = L1
    elif side in ['l','L',0]:   upto = L2
    else:                       upto = 0

    if np.all( [len(dp) == 1 for dp in pepx_u.phys_bonds.flat] ):
        boundaries = get_boundaries(pepx_u,pepx_d,side,upto,XMAX=XMAX,get_err=get_err)
    elif np.all( [len(dp) == 2 for dp in pepx_u.phys_bonds.flat] ):
        # flatten bonds 
        pepx_uf = PEPX.flatten(pepx_u)  #PEPX.transposeUD(pepx_u))
        pepx_df = PEPX.flatten(pepx_d)  #PEPX.transposeUD(pepx_d))
        boundaries = get_boundaries(pepx_uf,pepx_df,side,upto,XMAX=XMAX,get_err=get_err)
    else:
        print 'in env_rho contract:  please use pepo or pep0'
        exit()
    
    if not get_err:   
        bound = boundaries[upto]

        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        # ntens = np.einsum('luur->lr',bound[0])

        # for i in range(1,len(bound)):
        #     b_ = np.einsum('ruus->rs',bound[i])         # U should be dim 1
        #     ntens = np.einsum('lr,rs->ls',ntens,b_)

        return ovlp
    else:
        bounds, errs = boundaries
        bound = bounds[upto]
       
        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        # ntens = np.einsum('luur->lr',bound[0])

        # for i in range(1,len(bound)):
        #     b_ = np.einsum('ruus->rs',bound[i])         # U should be dim 1
        #     ntens = np.einsum('lr,rs->ls',ntens,b_)

        return ovlp, errs


def get_sub_ovlp(sub_pepx_u, sub_pepx_d, bounds, side=None, XMAX=100, get_err=False):

    L1, L2 = sub_pepx_u.shape

    if side is None:
        if L1 >= L2:  side = 'I'
        else:         side = 'L'

    if   side in ['i','I',1]:   upto = L1
    elif side in ['l','L',0]:   upto = L2
    else:                       upto = 0

    bL, bI, bO, bR = bounds
    if not get_err:
        if side   in ['i','I',1]:
            bounds = get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bI,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bO)
        elif side in ['o','O',2]:
            bounds = get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bO,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bI)
        elif side in ['l','L',0]:
            bounds = get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bL,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bL)
        elif side in ['r','R',3]:
            bounds = get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bR,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bR)

        return ovlp

    else:
        if side   in ['i','I',1]:
            bounds,errs= get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bI,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bO)
        elif side in ['o','O',2]:
            bounds,errs= get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bO,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bI)
        elif side in ['l','L',0]:
            bounds,errs= get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bL,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bL)
        elif side in ['r','R',3]:
            bounds,errs= get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bR,XMAX=XMAX,get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bR)

        return ovlp, errs


############################################
####          embedded methods          ####
############################################

def embed_sites_norm(sub_pepx,envs_list,side='L',XMAX=100,get_errs=False):
    ''' get norm of sub_pepx embedded in env '''

    # # out =  embed_sites_ovlp( np.conj(PEPX.transposeUD(sub_pepx)), sub_pepx, envs_list, side, XMAX, get_errs)
    # out =  embed_sites_ovlp( np.conj(sub_pepx), sub_pepx, envs_list, side, XMAX, get_errs)
    # if get_errs:
    #     norm2, errs = out
    #     return np.sqrt(norm2), errs
    # else:
    #     return np.sqrt(out)

    out = embed_sites_ovlp( np.conj(sub_pepx), sub_pepx, envs_list)
    return np.sqrt(out)

# def embed_sites_ovlp(sub_pepx_u, sub_pepx_d, envs_list, side='L',XMAX=100,get_errs=False):
#     ''' get ovlp of sub_pepx_u, sub_pepx_d embedded in env '''
# 
#     L1, L2 = sub_pepx_u.shape
#     bL, bI, bO, bR = envs_list
# 
#     embedded_pepx = sub_pepx_d.copy()
# 
#     # print 'embed sites ovlp'
#     # print [m.shape for m in bL]
#     # print [m.shape for m in bI]
#     # print [m.shape for m in bO]
#     # print [m.shape for m in bR]
#     # print [m.shape for idx,m in np.ndenumerate(sub_pepx_u)]
#  
#     cum_err = 0
#     if   side in ['l','L',0]:
#         sb = bL
#         for j in range(L2):
#             subbound, err = get_next_subboundary_L(sb,bI[j],sub_pepx_u[:,j],sub_pepx_d[:,j],bO[j],XMAX=XMAX)
#             sb = subbound
#             cum_err += err
#         ovlp = contract_2_bounds(sb,bR)
# 
#     elif side in ['r','R',3]:
#         sb = bR
#         for j in range(L2)[::-1]:
#             subbound, err = get_next_subboundary_R(sb,bI[j],sub_pepx_u[:,j],sub_pepx_d[:,j],bO[j],XMAX=XMAX)
#             sb = subbound
#             cum_err += err
#         ovlp = contract_2_bounds(sb,bL)
# 
#     elif side in ['i','I',1]:
#         sb = bI
#         for i in range(L1):
#             subbound, err = get_next_subboundary_I(sb,bL[i],sub_pepx_u[i,:],sub_pepx_d[i,:],bR[i],XMAX=XMAX)
#             sb = subbound
#             cum_err += err
#         ovlp = contract_2_bounds(sb,bO)
# 
#     elif side in ['o','O',2]:
#         sb = bO
#         for i in range(L1)[::-1]:
#             subbound, err = get_next_subboundary_O(sb,bL[i],sub_pepx_u[i,:],sub_pepx_d[i,:],bR[i],XMAX=XMAX)
#             sb = subbound
#             cum_err += err
#         ovlp = contract_2_bounds(sb,bI)
# 
# 
#     if get_errs:    return ovlp, cum_err
#     else:           return ovlp


def embed_sites_ovlp(sub_peps_u, sub_peps_d, envs_list, side='L',XMAX=100,get_errs=False):

    return embed_sites(sub_peps_u, sub_peps_d, envs_list, (0,0), 'oo')
 

##################################
# embedding:  keeping sites  #
##################################


def embed_sites_xo(sub_peps_u, sub_peps_d, envs, x_idx):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    return embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, 'xo')


# @profile
def embed_sites_xx(sub_peps_u, sub_peps_d, envs, x_idx):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''
    return embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, 'xx')


def embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, idx_key):

    # assume sum(shape(bra)) < sum(shape(ket))

    L1,L2 = sub_peps_u.shape
    bL, bI, bO, bR = envs

    if (L1,L2) == (1,1):
        
        su00 = sub_peps_u[0,0]
        sd00 = sub_peps_d[0,0]

        if idx_key == 'xx':
            raise(NotImplementedError)
        elif idx_key == 'xo':
            raise(NotImplementedError)
        elif idx_key == 'oo':
            norm = np.einsum('xlLy,yiIz->xlLiIz',bL[0],bI[0])
            norm = np.einsum('xlLiIz,liord->xLIordz',norm,su00)
            norm = np.einsum('xLIordz,LIORd->xoOrRz',norm,sd00)
            norm = np.einsum('xoOrRz,xoOw->wrRz',norm,bO[0])
            norm = np.einsum('wrRz,zrRw->',norm,bR[0])
            return norm


    elif (L1,L2) == (1,2):   # horizontal trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = sub_peps_u[0,1]
            s2d = sub_peps_d[0,1]

            # env_block2 = np.einsum('wiIx,liord->wlIorxd', bI[1],s2u)
            # env_block2 = np.einsum('wlIorxd,LIORd->wlLoOrRx',env_block2,s2d)        # inner boundary
            # env_block2 = np.einsum('wlLoOrRx,xrRy->wlLoOy',env_block2, bR[0])     # right boundary
            # env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2, bO[1])         # outer boundary
            env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
            env_block2 = np.einsum('wiIrRy,liord->wlIoRdy',env_block2,s2u)
            env_block2 = np.einsum('wlIoRdy,LIORd->wlLoOy',env_block2,s2d)
            env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2,bO[1])
    
            ## site 1 boundary
            s1u = sub_peps_u[0,0]
            s1d = sub_peps_d[0,0]

            if idx_key == 'xx':    ## metric
                # env_xx = np.einsum('xiIy,yrRz->xiIrRz',bI[0],env_block2)
                # env_xx = np.einsum('wiIrRz,xoOz->wiIoOrRx',env_xx,bO[0])
                # env_xx = np.einsum('wlLx,wiIoOrRx->lLiIoOrR',bL[0],env_xx)
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,yoOz->wlLiIoOz',env_block1,bO[0])
                return np.einsum('wlLiIoOz,wrRz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':  ## grad (missing bra)
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,LIORd->wliORdy',env_block1,s1d)
                env_block1 = np.einsum('wliORdy,yoOz->wlioRdz',env_block1,bO[0])
                return np.einsum('wlioRdz,wrRz->liord',env_block1,env_block2)
            elif idx_key == 'oo':  ## norm
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)
                env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
                env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])
                return np.einsum('wrRz,wrRz->',env_block1,env_block2)
    

        elif x_idx == (0,1):
            ## update site 1
            s1u = sub_peps_u[0,0]
            s1d = sub_peps_d[0,0]
        
            # env_block1 = np.einsum('xiIw,liord->wlIorxd',bI[0],s1u)
            # env_block1 = np.einsum('wlIorxd,LIORd->wlLoOrRx',env_block1,s1d)     # inner boundary
            # env_block1 = np.einsum('wlLoOrRx,xlLy->woOrRy',env_block1,bL[0])     # left boundary
            # env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])         # outer boundary
            env_block1 = np.einsum('xlly,xiiw->wlliiy',bl[0],bi[0])
            env_block1 = np.einsum('wlliiy,liord->wliordy',env_block1,s1u)
            env_block1 = np.einsum('wliordy,liord->woorry',env_block1,s1d)
            env_block1 = np.einsum('woorry,yooz->wrrz',env_block1,bo[0])


            ## site 2 boundary
            s2u = sub_peps_u[0,1]
            s2d = sub_peps_d[0,1]

            if idx_key == 'xx':
                # env_xx = np.einsum('xiIy,xlLw->wlLiIy',bI[1],env_block1)
                # env_xx = np.einsum('wlLiIy,woOz->lLiIoOyz',env_xx,bO[1])
                # env_xx = np.einsum('lLiIoOyz,yrRz->lLiIoOrR',env_xx,bR[0])
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('wiIrRy,zoOy->wiIoOrRz',env_block2,bO[1])
                return np.einsum('wlLz,wiIoOrRz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('LIORd,wiIrRy->wLiOrdy',s2d,env_block2)
                env_block2 = np.einsum('wLiOrdy,zoOy->wLiordz',env_block2,bO[1])
                return np.einsum('wlLz,wLiordz->liord',env_block1,env_block2)
            elif idx_key == 'oo':
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('liord,wiIrRy->wlIoRdy',s2u,env_block2)
                env_block2 = np.einsum('LIORd,wlIoRdy->wlLoOy',s2d,env_block2)
                env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2,bO[1])
                return np.einsum('wlLz,wlLz->',env_block1,env_block2)

        else:  raise (IndexError)
        

    elif (L1,L2) == (2,1):   # vertical trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = sub_peps_u[1,0]
            s2d = sub_peps_d[1,0]

            # env_block2 = np.einsum('wlLx,liord->wLiorxd',bL[1],s2u)
            # env_block2 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block2,s2d)     # left boundary
            # env_block2 = np.einsum('xoOy,wiIoOrRx->wiIrRy',bO[0],env_block2)     # outer boundary
            # env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])         # right boundary
            env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
            env_block2 = np.einsum('wlLoOy,liord->wLiOrdy',env_block2,s2u)
            env_block2 = np.einsum('wLiOrdy,LIORd->wiIrRy',env_block2,s2d)
            env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])

            ## site 1 boundary
            s1u = sub_peps_u[0,0]
            s1d = sub_peps_d[0,0]
    
            if idx_key == 'xx':
                # env_xx = np.einsum('wlLx,xoOy->wlLoOy',bL[0],env_block2)
                # env_xx = np.einsum('wlLoOy,zrRy->wlLoOrRz',env_xx,bR[0])
                # env_xx = np.einsum('wlLoOrRz,wiIz->lLiIoOrR',env_xx,bI[0])
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,yrRz->wlLiIrRz',env_block1,bR[0])
                return np.einsum('wlLiIrRz,woOz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,LIORd->wliORdy',env_block1,s1d)
                env_block1 = np.einsum('wliORdy,yrRz->wliOrdz',env_block1,bR[0])
                return np.einsum('wliOrdz,woOz->liord',env_block1,env_block2)
            elif idx_key == 'oo':
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)
                env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
                env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])
                return np.einsum('woOz,woOz->',env_block1,env_block2)
    
        elif x_idx == (1,0):
            ## update site 1
            s1u = sub_peps_u[0,0]
            s1d = sub_peps_d[0,0]
        
            # env_block1 = np.einsum('xlLw,liord->wLiorxd',bL[0],s1u)
            # env_block1 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block1,s1d)     # left boundary
            # env_block1 = np.einsum('xiIy,wiIoOrRx->woOrRy',bI[0],env_block1)     # inner boundary
            # env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])         # right boundary
            env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)   # assume sum(shape(s1u)) < sum(sh(s2u))
            env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
            env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])

            ## site 2 boundary
            s2u = sub_peps_u[1,0]
            s2d = sub_peps_d[1,0]
    
            if idx_key == 'xx':
                # env_xx = np.einsum('wlLx,wiIy->xlLiIy',bL[1],env_block1)
                # env_xx = np.einsum('xlLiIy,yrRz->xlLiIrRz',env_xx,bR[1])
                # env_xx = np.einsum('xlLiIrRz,xoOz->lLiIoOrR',env_xx,bO[0])
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,zrRy->wlLoOrRz',env_block2,bR[1])
                return np.einsum('wiIz,wlLoOrRz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,LIORd->wlIoRdy',env_block2,s2d)
                env_block2 = np.einsum('wlIoRdy,zrRy->wlIordz',env_block2,bR[1])
                return np.einsum('wiIz,wlIordz->liord',env_block1,env_block2)
            elif idx_key == 'oo':
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,liord->wLiOrdy',env_block2,s2u)  # assume sum(shape(s1u)) < sum(sh(s2u)
                env_block2 = np.einsum('wLiOrdy,LIORd->wiIrRy',env_block2,s2d)
                env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])
                return np.einsum('wiIz,wiIz->',env_block1,env_block2)

        else:  raise (IndexError)
     
   
    elif (L1,L2) == (2,2):  # LR/square trotter step

        def get_bL10(bL1,tens10_u,tens10_d,bO0):
            # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])
            if np.sum(tens10_u.shape) > np.sum(tens10_d.shape):
                # bL10 = np.einsum('wlLx,LIORd->wlIORdx',bL1,tens10_d)
                # bL10 = np.einsum('wlIORdx,liord->wiIoOrRx',bL10,tens10_u)
                # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                bL10 = np.einsum('wlLoOy,LIORd->wlIoRyd',bL10,tens10_d)
                bL10 = np.einsum('wlIoRyd,liord->wiIrRy',bL10,tens10_u)
            else:
                # bL10 = np.einsum('wlLx,liord->wLiordx',bL1,tens10_u)
                # bL10 = np.einsum('wLiordx,LIORd->wiIoOrRx',bL10,tens10_d)
                # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                bL10 = np.einsum('wlLoOy,liord->wLiOryd',bL10,tens10_u)
                bL10 = np.einsum('wLiOryd,LIORd->wiIrRy',bL10,tens10_d)
            return bL10
 
        def get_bL11(bO1,tens11_u,tens11_d,bR1):
            # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])
            if np.sum(tens11_u.shape) > np.sum(tens11_d.shape):
                # bL11 = np.einsum('xoOy,LIORd->xLIoRyd',bO1,tens11_d)
                # bL11 = np.einsum('xLIoRyd,liord->xlLiIrRy',bL11,tens11_u)
                # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                bL11 = np.einsum('xoOrRz,LIORd->xLIorzd',bL11,tens11_d)
                bL11 = np.einsum('xLIorzd,liord->xlLiIz',bL11,tens11_u)
            else:
                # bL11 = np.einsum('xoOy,liord->xliOryd',bO1,tens11_u)
                # bL11 = np.einsum('xliOryd,LIORd->xlLiIrRy',bL11,tens11_d)
                # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                bL11 = np.einsum('xoOrRz,liord->xliORzd',bL11,tens11_u)
                bL11 = np.einsum('xliORzd,LIORd->xlLiIz',bL11,tens11_d)
            return bL11

        def get_bL01(bI1,tens01_u,tens01_d,bR0):
            # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])
            if np.sum(tens01_u.shape) > np.sum(tens01_d.shape):
                # bL01 = np.einsum('xiIy,LIORd->xLiORyd',bI1,tens01_d)
                # bL01 = np.einsum('xLiORyd,liord->xlLoOrRy',bL01,tens01_u)
                # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                bL01 = np.einsum('xiIrRz,LIORd->xLiOrzd',bL01,tens01_d)
                bL01 = np.einsum('xLiOrzd,liord->xlLoOz',bL01,tens01_u)
            else:
                # bL01 = np.einsum('xiIy,liord->xlIoryd',bI1,tens01_u)
                # bL01 = np.einsum('xlIoryd,LIORd->xlLoOrRy',bL01,tens01_d)
                # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                bL01 = np.einsum('xiIrRz,liord->xlIoRzd',bL01,tens01_u)
                bL01 = np.einsum('xlIoRzd,LIORd->xlLoOz',bL01,tens01_d)
            return bL01

        def get_bL00(bL0,tens00_u,tens00_d,bI0):
            # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])
            if np.sum(tens00_u.shape) > np.sum(tens00_d.shape):
                # bL00 = np.einsum('ylLx,LIORd->xlIORyd',bL0,tens00_d)
                # bL00 = np.einsum('xlIORyd,liord->xiIoOrRy',bL00,tens00_u)
                # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                bL00 = np.einsum('xlLiIz,LIORd->xliORzd',bL00,tens00_d)
                bL00 = np.einsum('xliORzd,liord->xoOrRz',bL00,tens00_u)
            else:
                # bL00 = np.einsum('ylLx,liord->xLioryd',bL0,tens00_u)
                # bL00 = np.einsum('xLioryd,LIORd->xiIoOrRy',bL00,tens00_d)
                # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                bL00 = np.einsum('xlLiIz,liord->xLIorzd',bL00,tens00_u)
                bL00 = np.einsum('xLIorzd,LIORd->xoOrRz',bL00,tens00_d)
            return bL00


        ### order contractions assuming 'rol' mpo connectivity ###
        ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
        ###                               10 < 00 ~ 11 < 01 for 3-body operator
        if x_idx == (0,0):

            # update other corners
            bL10 = get_bL10(bL[1],sub_peps_u[1,0],sub_peps_d[1,0],bO[0])
            bL11 = get_bL11(bO[1],sub_peps_u[1,1],sub_peps_d[1,1],bR[1])
            bL01 = get_bL01(bI[1],sub_peps_u[0,1],sub_peps_d[0,1],bR[0])

            ## 10 -> 11 -> 01
            bLs = np.einsum('wiIrRx,xrRoOy->wiIoOy',bL10,bL11)
            bLs = np.einsum('wiIoOy,zlLoOy->wiIlLz',bLs, bL01)

            # # corner w/o site
            if idx_key == 'xx':    ## metric
                # env_xx = np.einsum('xiIy,woOrRy->xwiIoOrR',bI[0],bLs)
                # env_xx = np.einsum('xlLw,xwiIoOrR->lLiIoOrR',bL[0],env_xx)
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                return np.einsum('xlLiIz,xoOrRz->lLiIoOrR',bL00,bLs)
            elif idx_key == 'xo':
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                bL00 = np.einsum('xlLiIz,LIORd->xliORdz',bL00,sub_peps_d[0,0])
                return np.einsum('xliORdz,xoOrRz->liord',bL00,bLs)
            elif idx_key == 'oo':
                bL00 = get_bL00(bL[0],sub_peps_u[0,0],sub_peps_d[0,0],bI[0])
                return np.einsum('xoOrRz,xoOrRz->',bL00,bLs)

        elif x_idx == (0,1):

            # update other corners
            bL10 = get_bL10(bL[1],sub_peps_u[1,0],sub_peps_d[1,0],bO[0])
            bL11 = get_bL11(bO[1],sub_peps_u[1,1],sub_peps_d[1,1],bR[1])
            bL00 = get_bL00(bL[0],sub_peps_u[0,0],sub_peps_d[0,0],bI[0])

            ## 00 -> 10 -> 11
            bLs = np.einsum('xoOrRw,xoOlLy->wrRlLy',bL00,bL10)
            bLs = np.einsum('wrRlLy,ylLiIz->wrRiIz',bLs ,bL11)

            # # corner w/o site
            if idx_key == 'xx':    ## metric
                # env_xx = np.einsum('wlLoOy,wiIx->lLiIoOxy',bLs,bI[1])
                # env_xx = np.einsum('lLiIoOxy,xrRy->lLiIoOrR',env_xx,bR[0])
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                return np.einsum('xiIrRz,xlLoOz->lLiIoOrR',bL01,bLs)
            elif idx_key == 'xo':  ## gradient
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                bL01 = np.einsum('xiIrRz,LIORd->xLiOrdz',bL01,sub_peps_d[0,1])
                return np.einsum('xlLoOz,xLiOrdz->liord',bLs,bL01)
            else:
                bL01 = get_bL01(bI[1],sub_peps_u[0,1],sub_peps_d[0,1],bR[0])
                return np.einsum('xlLoOz,xlLoOz->',bLs,bL01)

        elif x_idx == (1,0):

            # update other corners
            bL00 = get_bL00(bL[0],sub_peps_u[0,0],sub_peps_d[0,0],bI[0])
            bL01 = get_bL01(bI[1],sub_peps_u[0,1],sub_peps_d[0,1],bR[0])
            bL11 = get_bL11(bO[1],sub_peps_u[1,1],sub_peps_d[1,1],bR[1])

            ## 00 -> 01 -> 11
            bLs = np.einsum('woOrRx,xrRiIy->woOiIy',bL00,bL01)
            bLs = np.einsum('woOiIy,zlLiIy->woOlLz',bLs, bL11)
 
            # # corner w/o site
            if idx_key == 'xx':
                # env_xx = np.einsum('wiIrRz,xoOz->wxiIoOrR',bLs,bO[0])
                # env_xx = np.einsum('wlLx,wxiIoOrR->lLiIoOrR',bL[1],env_xx)
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                return np.einsum('xlLoOz,xiIrRz->lLiIoOrR',bL10,bLs)
            elif idx_key == 'xo':
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                bL10 = np.einsum('xlLoOz,LIORd->xlIoRdz',bL10,sub_peps_d[1,0])
                return np.einsum('xlIoRdz,xiIrRz->liord',bL10,bLs)
            elif idx_key == 'oo':
                bL10 = get_bL10(bL[1],sub_peps_u[1,0],sub_peps_d[1,0],bO[0])
                return np.einsum('xiIrRz,xiIrRz->',bL10,bLs)

        elif x_idx == (1,1):

            # update other corners
            bL01 = get_bL01(bI[1],sub_peps_u[0,1],sub_peps_d[0,1],bR[0])
            bL00 = get_bL00(bL[0],sub_peps_u[0,0],sub_peps_d[0,0],bI[0])
            bL10 = get_bL10(bL[1],sub_peps_u[1,0],sub_peps_d[1,0],bO[0])

            ## 00 -> 10 -> 01
            bLs = np.einsum('xiIrRw,xiIlLy->wrRlLy',bL10,bL00)
            bLs = np.einsum('wrRlLy,ylLoOz->wrRoOz',bLs, bL01)

            # # corner w/o site
            if idx_key == 'xx':
                # env_xx = np.einsum('wlLiIy,woOx->lLiIoOyx',bLs,bO[1])
                # env_xx = np.einsum('lLiIoOyx,yrRx->lLiIoOrR',env_xx,bR[1])
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                return np.einsum('xlLiIz,xoOrRz->lLiIoOrR',bLs,bL11)
            elif idx_key == 'xo':
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                bL11 = np.einsum('xoOrRz,LIORd->xLIordz',bL11,sub_peps_d[1,1])
                return np.einsum('xlLiIz,xLIordz->liord',bLs,bL11)
            elif idx_key == 'oo':
                bL11 = get_bL11(bO[1],sub_peps_u[1,1],sub_peps_d[1,1],bR[1])
                return np.einsum('xlLiIz,xlLiIz->',bLs,bL11)


        else:  raise (IndexError)

    else:
        raise (NotImplementedError)
           



def embed_sites_xo_bm(sub_peps_u, sub_peps_d, envs, x_idx, XMAX=100, update_conn=None):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    L1,L2 = sub_peps_u.shape
    bL, bI, bO, bR = envs

    env_xx = embed_sites_xx_bm(sub_peps_u,sub_peps_d,envs,x_idx,XMAX,update_conn)
    env_xo = np.einsum('lLiIoOrR,LIORd->liord',env_xx,sub_peps_d[x_idx])

    return env_xo

def embed_sites_xx_bm(sub_peps_u, sub_peps_d, envs, x_idx, XMAX=100, update_conn=None):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''
    L1,L2 = sub_peps_u.shape
    bL, bI, bO, bR = envs


    if (L1,L2) == (1,2) or (L1,L2) == (2,1):   # horizontal/vertical trotter step

        env_xx = embed_sites_xx(sub_peps_u,sub_peps_d,envs,x_idx)

    elif (L1,L2) == (2,2):  # LR/square trotter step

        if update_conn in ['i','o']:  use_vertenv = True
        else:                         use_vertenv = False

        if x_idx == (0,0):

            if use_vertenv:
                bR_, err = get_next_subboundary_R(bR,bI[1],sub_peps_u[:,1],sub_peps_d[:,1],bO[1],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[:,:1],sub_peps_d[:,:1],[bL,bI[:1],bO[:1],bR_],(0,0))
            else:
                bO_, err = get_next_subboundary_O(bO,bL[1],sub_peps_u[1,:],sub_peps_d[1,:],bR[1],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[:1,:],sub_peps_d[:1,:],[bL[:1],bI,bO_,bR[:1]],(0,0))

        elif x_idx == (0,1):

            if use_vertenv:
                bL_, err = get_next_subboundary_L(bL,bI[0],sub_peps_u[:,0],sub_peps_d[:,0],bO[0],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[:,1:],sub_peps_d[:,1:],[bL_,bI[1:],bO[1:],bR],(0,0))
            else:
                bO_, err = get_next_subboundary_O(bO,bL[1],sub_peps_u[1,:],sub_peps_d[1,:],bR[1],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[:1,:],sub_peps_d[:1,:],[bL[:1],bI,bO_,bR[:1]],(0,1))

        elif x_idx == (1,0):

            if use_vertenv:
                bR_, err = get_next_subboundary_R(bR,bI[1],sub_peps_u[:,1],sub_peps_d[:,1],bO[1],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[:,:1],sub_peps_d[:,:1],[bL,bI[:1],bO[:1],bR_],(1,0))
            else:
                bI_, err = get_next_subboundary_I(bI,bL[0],sub_peps_u[0,:],sub_peps_d[0,:],bR[0],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[1:,:],sub_peps_d[1:,:],[bL[1:],bI_,bO,bR[1:]],(0,0))

        elif x_idx == (1,1):

            if use_vertenv:
                bL_, err = get_next_subboundary_L(bL,bI[0],sub_peps_u[:,0],sub_peps_d[:,0],bO[0],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[:,1:],sub_peps_d[:,1:],[bL_,bI[1:],bO[1:],bR],(1,0))
            else:
                bI_, err = get_next_subboundary_I(bI,bL[0],sub_peps_u[0,:],sub_peps_d[0,:],bR[0],XMAX)
                env_xx = embed_sites_xx(sub_peps_u[1:,:],sub_peps_d[1:,:],[bL[1:],bI_,bO,bR[1:]],(0,1))

        else:  raise (IndexError)

    else:
        raise (NotImplementedError)
           
    return env_xx



##################################
# embedding: saving environment  #
##################################

def build_env(envs,ensure_pos=False):

    bL, bI, bO, bR = envs
    L1 = len(bL)
    L2 = len(bI)

    env = bL[0]
    for b in bL[1:]:
        env = np.einsum('l...r,ruds->l...uds',env,b)

    for b in bI:
        env = np.einsum('l...s,ludr->r...uds',env,b)

    for b in bO:
        env = np.einsum('l...r,ruds->l...uds',env,b)

    for b in bR[:-1]:
        env = np.einsum('l...r,luds->s...udr',env,b)
    env = np.einsum('l...r,ludr->...ud',env,bR[-1])   # close the loop

    # env:  'L1, L2, ..., I1, I2, ..., O1, O2, ..., R1, R2...'

    ### ensure positivity
    if ensure_pos:

        axT = np.reshape( np.arange(env.ndim).reshape(-1,2).T, -1 )
        axT_inv = np.argsort(axT)
        sqdim = int(np.sqrt(np.prod(env.shape)))

        envIO = env.transpose(axT)
        block_env = envIO.reshape(sqdim,sqdim)

        ### Reza's method??   sqrt(M^* M)
        # b2 = np.dot(np.conj(block_env.T), block_env)
        
        u,s,vt = decompose_env(env)
        u = u.reshape(-1,len(s))
        vt = vt.reshape(len(s),-1)
        sqrt_b2 = np.dot(np.conj(vt.T), tf.dMult('DM',s,vt))  ### old 08/14/19
        # sqrt_b2 = np.dot(vt,tf.dMult('DM',s,np.conj(vt.T)))
        env_pos = np.reshape(sqrt_b2, envIO.shape).transpose(axT_inv)

        print 'build env ensure pos diff', np.linalg.norm(env_pos - env), np.linalg.norm(env)
        eD,eV = np.linalg.eig(block_env)
        print 'env eigvals',np.max(eD), np.min(eD)


        ### Lubasch's method
        # # block_sym = (block_env + np.conj(block_env.T))/2
        # # evals, evecs = np.linalg.eig(block_sym)

        # # evals = np.where(evals > 0, evals, 0*evals)
        # # # block_pos = np.dot(np.conj(evecs.T), tf.dMult('DM',evals,evecs))  ### old 08/14/19
        # # block_pos = np.dot(evecs.T, tf.dMult('DM',evals,np.conj(evecs.T)))
        # # env_pos = np.reshape(block_pos, envIO.shape).transpose(axT_inv)

        # print 'positive env difference', np.linalg.norm(env_pos-env)

        env = env_pos

    
    return env


def embed_sites_xo_env(sub_peps_u, sub_peps_d, env_tens, x_idx):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    L1,L2 = sub_peps_u.shape

    env_xx = embed_sites_xx_env(sub_peps_u,sub_peps_d,env_tens,x_idx)
    env_xo = np.einsum('lLiIoOrR,LIORd->liord',env_xx,sub_peps_d[x_idx])

    return env_xo


def embed_sites_xx_env(sub_peps_u, sub_peps_d, env_tens, x_idx):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''
    L1,L2 = sub_peps_u.shape

    if (L1,L2) == (1,2):   # horizontal trotter step

        # envblock: L1-I1-I2-O1-O2-R
    
        if x_idx == (0,0):
            ## update site 2
            s2u = sub_peps_u[0,1]
            s2d = sub_peps_d[0,1]

            env_block2 = np.einsum('lLiIjJoOpPrR,mjprd->lLiIJoOPmRd',env_tens,s2u)
            env_xx = np.einsum('lLiIJoOPmRd,MJPRd->lLiIoOmM',env_block2, s2d)
    
        elif x_idx == (0,1):
            ## update site 1
            s1u = sub_peps_u[0,0]
            s1d = sub_peps_d[0,0]
        
            env_block1 = np.einsum('lLiIjJoOpPrR,liomd->mLIjJOpPrRd',env_tens,s1u)
            env_xx = np.einsum('mLIjJOpPrRd,LIOMd->mMjJpPrR',env_block1, s1d)
    
        else:  raise (IndexError)
        

    elif (L1,L2) == (2,1):   # vertical trotter step
    
        # envblock:  L1-L2-I1-O2-R1-R2

        if x_idx == (0,0):
            ## update site 2
            s2u = sub_peps_u[1,0]
            s2d = sub_peps_d[1,0]
    
            env_block2 = np.einsum('lLmMiIoOrRsS,mjosd->lLMiIjOrRSd',env_tens,s2u)
            env_xx = np.einsum('lLMiIjOrRSd,MJOSd->lLiIjJrR',env_block2,s2d)
    
        elif x_idx == (1,0):
            ## update site 1
            s1u = sub_peps_u[0,0]
            s1d = sub_peps_d[0,0]

            env_block1 = np.einsum('lLmMiIoOrRsS,lijrd->LmMIjoORsSd',env_tens,s1u)
            env_xx = np.einsum('LmMIjoORsSd,LIJRd->mMjJoOsS',env_block1,s1d)

        else:  raise (IndexError)
     
   
    elif (L1,L2) == (2,2):  # LR/square trotter step

        # env tens:  'L1-L2 - I1-I2 - O1-O2 - R1-R2'

        if x_idx == (0,0):

            env01 = np.einsum('lLmMiIjJoOpPrRsS,njqrd->lLmMiIqJoOpPnRsSd',env_tens,sub_peps_u[0,1])
            env01 = np.einsum('lLmMiIqJoOpPnRsSd,NJQRd->lLmMiIqQoOpPnNsS',env01,sub_peps_d[0,1])

            env11 = np.einsum('lLmMiIjJoOpPrRsS,njpsd->lLmMiIJoOPrRnSd',env01,sub_peps_u[1,1])
            env11 = np.einsum('lLmMiIJoOPrRnSd,NJPSd->lLmMiIoOrRnN',env11,sub_peps_d[1,1])

            env_xx = np.einsum('lLmMiIoOrRsS,mjosd->lLMiIjOrRSd',env11,sub_peps_u[1,0])
            env_xx = np.einsum('lLMiIjOrRSd,MJOSd->lLiIjJrR',env_xx,sub_peps_d[1,0])

        elif x_idx == (0,1):

            env00 = np.einsum('lLmMiIjJoOpPrRsS,liqtd->tLmMqIjJoOpPrRsSd',env_tens,sub_peps_u[0,0])
            env00 = np.einsum('tLmMqIjJoOpPrRsSd,LIQTd->tTmMqQjJoOpPrRsS',env00,sub_peps_d[0,0])
 
            env10 = np.einsum('lLmMiIjJoOpPrRsS,miotd->lLtMIjJOpPrRsSd',env00,sub_peps_u[1,0])
            env10 = np.einsum('lLtMIjJOpPrRsSd,MIOTd->lLtTjJpPrRsS',env10,sub_peps_d[1,0])

            env_xx = np.einsum('lLmMiIoOrRsS,mjosd->lLMiIjOrRSd',env10,sub_peps_u[1,1])
            env_xx = np.einsum('lLMiIjOrRSd,MJOSd->lLiIjJrR',env_xx,sub_peps_d[1,1])

        elif x_idx == (1,0):

            env01 = np.einsum('lLmMiIjJoOpPrRsS,njqrd->lLmMiIqJoOpPnRsSd',env_tens,sub_peps_u[0,1])
            env01 = np.einsum('lLmMiIqJoOpPnRsSd,NJQRd->lLmMiIqQoOpPnNsS',env01,sub_peps_d[0,1])

            env11 = np.einsum('lLmMiIjJoOpPrRsS,njpsd->lLmMiIJoOPrRnSd',env01,sub_peps_u[1,1])
            env11 = np.einsum('lLmMiIJoOPrRnSd,NJPSd->lLmMiIoOrRnN',env11,sub_peps_d[1,1])

            env_xx = np.einsum('lLmMiIoOrRsS,lijrd->LmMjIoORsSd',env11,sub_peps_u[0,0])
            env_xx = np.einsum('LmMjIoORsSd,LIJRd->mMjJoOsS',env_xx,sub_peps_d[0,0])

        elif x_idx == (1,1):

            env00 = np.einsum('lLmMiIjJoOpPrRsS,liqtd->tLmMqIjJoOpPrRsSd',env_tens,sub_peps_u[0,0])
            env00 = np.einsum('tLmMqIjJoOpPrRsSd,LIQTd->tTmMqQjJoOpPrRsS',env00,sub_peps_d[0,0])
 
            env10 = np.einsum('lLmMiIjJoOpPrRsS,miotd->lLtMIjJOpPrRsSd',env00,sub_peps_u[1,0])
            env10 = np.einsum('lLtMIjJOpPrRsSd,MIOTd->lLtTjJpPrRsS',env10,sub_peps_d[1,0])

            env_xx = np.einsum('lLmMiIoOrRsS,lijrd->LmMjIoORsSd',env10,sub_peps_u[0,1])
            env_xx = np.einsum('LmMjIoORsSd,LIJRd->mMjJoOsS',env_xx,sub_peps_d[0,1])

        else:  raise (IndexError)

    else:
        raise (NotImplementedError)
           
    return env_xx


def embed_sites_norm_env(sub_pepx,env_tens,x_idx=(0,0)):
    ''' get norm of sub_pepx embedded in env '''

    out =  embed_sites_ovlp_env( np.conj(PEPX.transposeUD(sub_pepx)),sub_pepx,env_tens,x_idx)
    return np.sqrt(out)


def embed_sites_ovlp_env(sub_pepx_u, sub_pepx_d, env_tens, x_idx=(0,0)):
    ''' get ovlp of sub_pepx_u, sub_pepx_d embedded in env '''

    env_xo = embed_sites_xo_env(sub_pepx_u, sub_pepx_d, env_tens, x_idx)
    ovlp = np.einsum('liord,liord->',env_xo,sub_pepx_u[x_idx])

    return ovlp


# def embed_peps_env(peps_list,op_conn,env_tens,ket=True):
#     ''' embed (bra) ket in the environment '''
# 
#     L = len(peps_list)
#     
#     if op_conn is None:
#         tens_list = peps_list    # assumes d_side = 'R' for axes
#         reorder = False
#     else:
#         tens_list, axT_invs = PEPX.connect_pepx_list(peps_list,connect_list,side='R')
#         reorder = True
# 
#     if ket:
#         env_block = env_tens
#     else:
#         axTb = np.array([[n+1, n] for n in range(0,env_tens.ndim,2)]).reshape(-1)
#         axTb_inv = np.argsort(axTb)   # should be the same as axTb
#         env_block = env_tens.transpose(axTb)
#        
# 
#     if env_tens.ndim == 2*(2*L+2):   # peps_list is a chain (eg. 2x1, 1x2, or 2x2 with QR)
# 
#         # left-most site
#         tens = tens_list[0]          # lxydr
#         axTb = (1,ind*2+3,ind*2+5)   # env is lLiIoOjJpP...rR 
#         env_block = np.einsum(env_block,tens,axes=(axTb,(0,1,2))
# 
#         for ind in range(len(tens_list)):
# 
#             tens = tens_list[ind]    # lxydr
#             axTs = range(tens.ndim)
#             
#             axTb = (-1,ind*2+1,ind*2+2)
#             env_block = np.einsum(env_block,tens,axes=(axTb,(0,1,2))
# 
#             env_block = np.einsum(env_block,axTb,env_tens,axTs)


### hard-coded implementation, 
def embed_peps_env(sub_peps,env_tens,ket=True):
    ''' embed (bra) ket in the environment '''
 
    L1,L2 = sub_peps.shape

    # take transpose if contracting in bra 
    if ket:
        env_block = env_tens
    else:
        axTb = np.array([[n+1, n] for n in range(0,env_tens.ndim,2)]).reshape(-1)
        axTb_inv = np.argsort(axTb)   # should be the same as axTb
        env_block = env_tens.transpose(axTb)

    # embedding
    if (L1,L2) == (1,2):   # horizontal trotter step

        # envblock: L1-I1-I2-O1-O2-R2
    
        env_block1 = np.einsum('lLiIjJoOpPrR,LIOMd->lMijJopPrRd',env_tens,sub_peps[0,0])
        env_ket = np.einsum('lMijJopPrRd,MJPRe->liodjper',env_block1,sub_peps[0,1])

        # output:  L1-I1-O1-d1-I2-O2-d2-R2


    elif (L1,L2) == (2,1):   # vertical trotter step
    
        # envblock:  L1-L2-I1-O2-R1-R2

        env_block1 = np.einsum('lLmMiIoOrRsS,LIJRd->lmMiJoOrsSd',env_tens,sub_peps[0,0])
        env_ket = np.einsum('lmMiJoOrsSd,MJOSe->irldsmeo',env_block1,sub_peps[1,0])

        # output:  I1-R1-L1-d1-R2-L2-d2-O2   (ie. rotate env ccw 90 degrees)

   
    elif (L1,L2) == (2,2):  # LR/square trotter step

        raise(NotImplementedError)



    return env_ket


# def add_rand_noise_env(sub_peps, noise_level=1, env=None):
#     ''' build random tensors such that w/ env yield a normalized state
#         add as noise by superposing with existing peps 
# 
#         noise_level = 1 means purely random tensors
#     '''
# 
#     rand_peps = sub_peps.copy()
#     for idx, m in np.ndenumerate(rand_peps):
#         rand_peps[idx] = np.random.random_sample(m.shape)
# 
#     if env is not None:
#         rand_norm = embed_sites_norm_env(rand_peps, env)
#         rand_peps = PEPX.mul(1./rand_norm, rand_peps)
# 
#     coeff1 = float(noise_level)
#     coeff2 = np.sqrt(1-noise_level**2)
# 
#     new_peps = PEPX.mul(coeff1, rand_peps) + PEPX.mul(coeff2, sub_peps)
#     new_norm = embed_sites_norm_env(new_peps,env)
# 
#     if np.abs(new_norm - 1) > 1.0e-3:
#         print 'WARNING:  error in rand noise statte norm', new_norm
#         # exit()
# 
#     return new_peps


# def add_rand_noise_env(sub_peps, noise_level=1, env=None, noise_shape=None):
#     ''' build random tensors such that w/ env yield a normalized state
#         add as noise by superposing with existing peps 
# 
#         noise_level = 1 means purely random tensors
#     '''
# 
#     if noise_shape is None:
#         rand_peps = sub_peps.copy()
#         for idx, m in np.ndenumerate(rand_peps):
#             rand_peps[idx] = np.random.random_sample(m.shape)
#     else:
#         tens_array = np.empty(noise_shape.shape)
#         for idx, sh in noise_shape:
#             tens_array[idx] = np.random.random.random_sample(sh)
#         rand_peps = PEPX.PEPX(tens_array)
# 
#     new_peps = PEPX.mul(noise_level, rand_peps) + sub_peps
#     if env is not None:      new_norm = embed_sites_norm_env(new_peps,env)
#     else:                    new_norm = PEPX.norm(new_peps)
#     new_peps = PEPX.mul(1./new_norm, new_peps)
# 
#     return new_peps
#  
# 
# def random_tens_env(shape_array,env=None):
#     ''' random sub pepx normalized wrt env if it is not None '''
# 
#     sub_pepx = PEPX.random_tens(shape_array,normalize=False)
#     if env is not None:
#         pepx_norm = embed_sites_norm_env(sub_pepx,env)
#         sub_pepx = PEPX.mul(1./pepx_norm, sub_pepx)
# 
#     print 'rand subpepx norm', [np.linalg.norm(m) for idx,m in np.ndenumerate(sub_pepx)]
#    
#     return sub_pepx


def env_order_io(env):
    
    axT = np.reshape( np.arange(env.ndim).reshape(-1,2).T, -1 )
    axT_inv = np.argsort(axT)
    sqdim = int(np.sqrt(np.prod(env.shape)))

    envIO = env.transpose(axT)

    return envIO, axT_inv


def decompose_env(env):
    ''' perform SVD / eval decomposition of env '''

    env_io, axT_inv = env_order_io(env)
    io_shape = env_io.shape
    io_dim  = env_io.ndim

    sqdim = int(np.sqrt(np.prod(env.shape)))
    block_env = env_io.reshape(sqdim,sqdim)

    u,s,vt = np.linalg.svd(block_env)

    u = u.reshape(io_shape[:io_dim/2]+(-1,))
    vt = vt.reshape((-1,)+io_shape[io_dim/2:])

    # u:  1234...-r
    # vt: r-1234...

    return u,s,vt


# def gauge_fix_metric(metric):
#     ''' QR decomposition of metric -> Q Q^T along all bond directions '''
#     
#     u,s,vt = decompose_env(metric)  
# 
#     print 'svd metric', np.linalg.norm(metric), s 
# 
#     metric_u = tf.dMult('MD',u,np.sqrt(s))
#     metric_d = tf.dMult('DM',np.sqrt(s),vt)
# 
#     gauge_u = metric_u.copy()
#     gauge_d = metric_d.copy()
#     # print 'decompose', len(s),gauge_u.shape, gauge_d.shape
# 
#     Rs = []
#     for xx in range(metric_u.ndim-1): 
#         iso_env = np.moveaxis(metric_u,xx,-1)
#         q,r = tf.qr(iso_env, '...,i')
#         Rs.append(r)
# 
#     return Rs
# 
# 
# def gauge_fix_env(env,sub_peps_u,sub_peps_d,op_conn):
# 
#     # env:  'L1, L2, ..., I1, I2, ..., O1, O2, ..., R1, R2...'
#     new_env = env.copy()
#     Ls = np.shape(sub_peps_u)
#     env_iRs = np.empty(Ls,dtype=np.object)
# 
#     for idx in np.ndindex(Ls):
# 
#         env_xx = embed_sites_xx_env(sub_peps_u,sub_peps_d,env,idx)
#         Rs = gauge_fix_metric(env_xx)
#         env_iRs[idx] = [np.linalg.inv(r) for r in Rs]
# 
#     if op_conn in ['r','R']:
# 
#         # (0,0)
#         iRs = env_iRs[0,0]   # l,i,o,r
#         new_env = np.einsum('xLiIjJoOpPrR,xl->lLiIjJoOpPrR',new_env,iRs[0])
#         new_env = np.einsum('lXiIjJoOpPrR,XL->lLiIjJoOpPrR',new_env,np.conj(iRs[0]))
#         new_env = np.einsum('lLxIjJoOpPrR,xi->lLiIjJoOpPrR',new_env,iRs[1])
#         new_env = np.einsum('lLiXjJoOpPrR,XI->lLiIjJoOpPrR',new_env,np.conj(iRs[1]))
#         new_env = np.einsum('lLiIjJxOpPrR,xo->lLiIjJoOpPrR',new_env,iRs[2])
#         new_env = np.einsum('lLiIjJoXpPrR,XO->lLiIjJoOpPrR',new_env,np.conj(iRs[2]))
# 
#         # (0,1)
#         iRs = env_iRs[0,1]   # l,i,o,r
#         new_env = np.einsum('lLiIxJoOpPrR,xj->lLiIjJoOpPrR',new_env,iRs[1])
#         new_env = np.einsum('lLiIjXoOpPrR,XJ->lLiIjJoOpPrR',new_env,np.conj(iRs[1]))
#         new_env = np.einsum('lLiIjJoOxPrR,xp->lLiIjJoOpPrR',new_env,iRs[2])
#         new_env = np.einsum('lLiIjJoOpXrR,XP->lLiIjJoOpPrR',new_env,np.conj(iRs[2]))
#         new_env = np.einsum('lLiIjJoOpPxR,xr->lLiIjJoOpPrR',new_env,iRs[3])
#         new_env = np.einsum('lLiIjJoOpPrX,XR->lLiIjJoOpPrR',new_env,np.conj(iRs[3]))
#    
#     elif op_conn in ['o','O']:
# 
#         # (0,0)
#         iRs = env_iRs[0,0]   # l,i,o,r
#         new_env = np.einsum('xLmMiIoOrRsS,xl->lLmMiIoOrRsS',new_env,iRs[0])
#         new_env = np.einsum('lXmMiIoOrRsS,XL->lLmMiIoOrRsS',new_env,np.conj(iRs[0]))
#         new_env = np.einsum('lLmMxIoOrRsS,xi->lLmMiIoOrRsS',new_env,iRs[1])
#         new_env = np.einsum('lLmMiXoOrRsS,XI->lLmMiIoOrRsS',new_env,np.conj(iRs[1]))
#         new_env = np.einsum('lLmMiIoOxRsS,xr->lLmMiIoOrRsS',new_env,iRs[3])
#         new_env = np.einsum('lLmMiIoOrXsS,XR->lLmMiIoOrRsS',new_env,np.conj(iRs[3]))
# 
#         # (1,0) 
#         iRs = env_iRs[1,0]   # l,i,o,r
#         new_env = np.einsum('lLxMiIoOrRsS,xm->lLmMiIoOrRsS',new_env,iRs[0])
#         new_env = np.einsum('lLmXiIoOrRsS,XM->lLmMiIoOrRsS',new_env,np.conj(iRs[0]))
#         new_env = np.einsum('lLmMiIxOrRsS,xo->lLmMiIoOrRsS',new_env,iRs[2])
#         new_env = np.einsum('lLmMiIoXrRsS,XO->lLmMiIoOrRsS',new_env,np.conj(iRs[2]))
#         new_env = np.einsum('lLmMiIoOrRxS,xs->lLmMiIoOrRsS',new_env,iRs[3])
#         new_env = np.einsum('lLmMiIoOrRsX,XS->lLmMiIoOrRsS',new_env,np.conj(iRs[3]))
# 
#     else:
#         raise(NotImplementedError)
# 
#     return new_env, env_iRs
# 
# 
# 
# def gauge_fix_sub(env_iRs,sub_peps,op_conn,inds_list):
#     ''' given inverted R matrices used to gauge the environment, return optimizied subpeps tensors with R^-1 
#         multiplied to it (so that now appropriate env is the non-gauged environment)
#     '''
#     # env:  'L1, L2, ..., I1, I2, ..., O1, O2, ..., R1, R2...'
#     Ls = np.shape(sub_peps)
#     new_peps = sub_peps.copy()
# 
#     for xx in range(len(inds_list)):
#         idx = (inds_list[0][xx],inds_list[1][xx])
#     
#         tens = new_peps[idx]
# 
#         skip_legs = []
#         try:                  skip_legs += [PEPX.leg2ind(op_conn[xx])]
#         except(IndexError):   pass
#         try:                  skip_legs += [PEPX.leg2ind(PEPX.opposite_leg(op_conn[xx-1]))]
#         except(IndexError):   pass
# 
#         for leg in range(4):
#             if leg in skip_legs:  continue
#             
#             tens = np.tensordot(tens,iRs[leg],axes=(leg,0))
#             tens = np.moveaxis(tens,-1,leg)
# 
#         new_peps[idx] = tens
# 
#     return new_peps

