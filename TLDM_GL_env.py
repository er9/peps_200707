import numpy as np
import time
import tens_fcts as tf

import MPX
import PEPX
import PEPX_GL
import PEPS_GL_env as ENV_GL
import TLDM_GL


''' builds environment for PEPS+ancilla (ie. two-layer density matrices)
   
    follows sqrt construction
'''


########################################################
####               get boundary MPOs                ####

''' get boundaries by contracting rows/cols of pepx 
    obtain list of boundary MPOs/MPSs,
    up to row/col "upto"
    returns err from compression of MPO/MPS if desired

    xxx  note:  bonds/lambdas for legs sticking out are not included
    note:  bonds/lambdas for legs sticking out are  included
'''
########################################################

#### grouping of gamma/lambda makes it look right canon? can we make code s.t. canonicalization step
#### is not necessary?

def get_next_boundary_O(tfd_row_u,tfd_row_d,boundary,XMAX=100):
    '''  get outside boundary (facing in)
         note:  peps_x_row is a 1 x L2 pepx_GL
         note:  this fct takes conj tranpose of tfd_row_u  so both inputs have vertical bonds (d,a)
    '''
    
    L2 = len(tfd_row_u)
    boundary_mpo = boundary.copy()

    for j in range(L2):

        ptens_d = PEPX_GL.get_site(tfd_row_d,j,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_row_u,j,op='sqrt')

        tens2 = np.einsum('liorda,xoOy->xliOyrda',ptens_u,boundary_mpo[j])
        tens2 = np.einsum('LIORda,xliOyrda->xlLiIyrR',ptens_d,tens2)
        boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

 
    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    if np.sum(err) > 1.0:
        print 'bound O large compression error', XMAX
        print [np.linalg.norm(m) for m in boundary_mpo]
        print err
        # exit()

    return boundary_mpo, err


def get_next_boundary_I(tfd_row_u,tfd_row_d,boundary,XMAX=100):
    ''' get inside boundary (facing out)
        note:  peps_x_row is 1-D
    '''

    L2 = len(tfd_row_u)
    boundary_mpo = boundary.copy()

    for j in range(L2):

        ptens_d = PEPX_GL.get_site(tfd_row_d,j,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_row_u,j,op='sqrt')

        tens2 = np.einsum('xiIy,liorda->xlIoyrda',boundary_mpo[j],ptens_u)
        tens2 = np.einsum('xlIoyrda,LIORda->xlLoOyrR',tens2,ptens_d)
        boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,0)


    if np.sum(err) > 1.0:
        print 'bound I large compression error', XMAX
        print [np.linalg.norm(m) for m in boundary_mpo]
        print err
        # exit()

    return boundary_mpo, err


def get_next_boundary_L(tfd_col_u,tfd_col_d,boundary,XMAX=100):
    ''' get left boundary (facing right)
        note:  peps_x_col is 1-D
    '''

    L1 = len(tfd_col_u)
    boundary_mpo = boundary.copy()

    for i in range(L1):

        ptens_d = PEPX_GL.get_site(tfd_col_d,i,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_col_u,i,op='sqrt')

        tens2 = np.einsum('xlLy,liorda->xLioryda',boundary_mpo[i],ptens_u)
        tens2 = np.einsum('xLioryda,LIORda->xiIrRyoO',tens2,ptens_d)
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')
    
    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err  = MPX.compress_reg(boundary_mpo,XMAX,0)

    return boundary_mpo, err


def get_next_boundary_R(tfd_col_u,tfd_col_d,boundary,XMAX=100):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''

    L1 = len(tfd_col_u)
    boundary_mpo = boundary.copy()

    for i in range(L1):

        ptens_d = PEPX_GL.get_site(tfd_col_d,i,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_col_u,i,op='sqrt')

        tens2 = np.einsum('xlLy,liorda->xLioryda',boundary_mpo[i],ptens_u)
        tens2 = np.einsum('xLioryda,LIORda->xiIrRyoO',tens2,ptens_d)
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')

    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    return boundary_mpo, err


#####################
#### full method ####
#####################

def get_boundaries(tfd_bra,tfd_ket,side,upto,init_bound=None,XMAX=100,get_err=False):
    ''' get_boundaries for pep0 (no dangling physical bonds, e.g. traced out) 
    '''

    L1, L2 = peps_u.shape

    if side in ['o','O',2]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]

        for i in range(L1-1,upto-1,-1):      # building envs from outside to inside
            boundary_mpo, err = get_next_boundary_O(tfd_bra[i,:],tfd_ket[i,:],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]
            
    elif side in ['i','I',1]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
     
        for i in range(upto):              # building envs from inside to outside
            boundary_mpo, err = get_next_boundary_I(tfd_bra[i,:],tfd_ket[i,:],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
    
    elif side in ['l','L',0]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L1) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(upto):              # building envs from left to right
            boundary_mpo, err = get_next_boundary_L(tfd_bra[:,j],tfd_ket[:,j],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
       
    elif side in ['r','R',3]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L1) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(L2-1,upto-1,-1):      # building envs from right to left
            boundary_mpo, err = get_next_boundary_R(tfd_bra[:,j],tfd_ket[:,j],envs[-1],XMAX)
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

    pepx_GL:  remaining/dangling PEPS bond does not have lambda applied to it

    returns e_mpo (an mpo/mps)
'''
#############################################################################
#############################################################################

# note:  ordering of grouped axes in subboundary is independent of ordering of axes in boundary
# because that connection (btwn sb and b) is  given by axis that is not grouped

def get_next_subboundary_O(env_mpo,boundL_tens,tfd_row_u,tfd_row_d,boundR_tens,XMAX=100):

    L2 = len(tfd_row_u)
    e_mpo = env_mpo.copy()

    for j in range(L2):

        ptens_d = PEPX_GL.get_site(tfd_row_d,j,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_row_u,j,op='sqrt')


        tens = np.einsum('LIORda,xoOy->LxIoRyda',ptens_d,e_mpo[j])
        tens = np.einsum('liorda,LxIoRyda->lLxiIrRy',ptens_u,tens)
        e_mpo[j] = tens  # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wlLx,lLxiIrRy->wiIrRy',boundL_tens,e_mpo[0])   # i(rRo) -- (lLx)oO(rRy)
    e_mpo[-1] = np.einsum('...rRy,zrRy->...z',e_mpo[-1],boundR_tens)      # (lLx)oO(rRy) -- i(lLo)

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')
    
    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,1)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0)

    return e_mpo, err


def get_next_subboundary_I(env_mpo,boundL_tens,tfd_row_u,tfd_row_d,boundR_tens,XMAX=100):

    L2 = len(tfd_row_u)
    e_mpo = env_mpo.copy()
    
    for j in range(L2):

        ptens_d = PEPX_GL.get_site(tfd_row_d,j,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_row_u,j,op='sqrt')

        tens = np.einsum('xiIy,liorda->xlIoyrda',e_mpo[j],ptens_u)
        tens = np.einsum('xlIoyrda,LIORda->xlLoOyrR',tens,ptens_d)
        e_mpo[j] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('xlLw,xlLoOyrR->woOyrR',boundL_tens,e_mpo[0])   # (irR)o -- (xlL)iI(yrR)
    e_mpo[-1] = np.einsum('...yrR,yrRz->...z',e_mpo[-1],boundR_tens)      # (xlL)iI(yrR) -- (ilL)o

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')
    
    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1)

    return e_mpo, err


def get_next_subboundary_L(env_mpo,boundI_tens,tfd_col_u,tfd_col_d,boundO_tens,XMAX=100):

    L1 = len(tfd_col_u)

    e_mpo = env_mpo.copy()
    
    for i in range(L1):

        ptens_d = PEPX_GL.get_site(tfd_col_d,i,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_col_u,i,op='sqrt')

        tens = np.einsum('xlLy,liorda->xiLryoda',e_mpo[i],ptens_u)
        tens = np.einsum('xiLryoda,LIORda->xiIrRyoO',tens,ptens_d)
        e_mpo[i] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('xiIw,xiIrRyoO->wrRyoO',boundI_tens,e_mpo[0])    # l(oOr) -- (xiI)rR(yoO)
    e_mpo[-1] = np.einsum('...yoO,yoOz->...z',e_mpo[-1],boundO_tens)       # (xiI)rR(yoO) -- (liI)r


    if L1 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')

    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1)

    if np.sum(err) > 1.0:
        print 'subL large compression error', XMAX
        print [np.linalg.norm(m) for m in e_mpo]
        print [np.linalg.norm(m) for m in env_mpo], np.linalg.norm(boundI_tens), np.linalg.norm(boundO_tens)
        print [np.linalg.norm(m) for m in peps_u_col]
        print [np.linalg.norm(m) for m in peps_d_col]
        print err
        # exit()

    return e_mpo, err


def get_next_subboundary_R(env_mpo,boundI_tens,tfd_col_u,tfd_col_d,boundO_tens,XMAX=100):

    L1 = len(tfd_col_u)
    e_mpo = env_mpo.copy()


    for i in range(L1):

        ptens_d = PEPX_GL.get_site(tfd_col_d,i,op='sqrt')
        ptens_u = PEPX_GL.get_site(tfd_col_u,i,op='sqrt')

        tens = np.einsum('LIORda,xrRy->xILryOda',ptens_d,e_mpo[i])
        tens = np.einsum('liorda,xILryOda->xiIlLyoO',ptens_u,tens)
        e_mpo[i] = tens    # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wiIx,xiIlLyoO->wlLyoO',boundI_tens,e_mpo[0])    # l(oOr) -- (iIx)lL(oOy)
    e_mpo[-1] = np.einsum('...yoO,zoOy->...z',e_mpo[-1],boundO_tens)       # (iIx)lL(oOy) -- l(iIr)

    if L1 > 1:    # if L1 == 1 don't need to do any reshaping
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')
    
    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,1)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0)

    return e_mpo, err


#####################
#### full method ####
#####################

def get_subboundaries(bound1,bound2,tfd_bra,tfd_ket,side,upto,init_sb=None,XMAX=100,get_errs=False):
    
    L1, L2 = peps_u_sub.shape  

    if side in ['o','O',2]:    # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(L1-1,upto-1,-1):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_O(envs[-1],bound1[i],tfd_bra[i,:],tfd_ket[i,:],bound2[i],XMAX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]

    elif side in ['i','I',1]:   # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(upto):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_I(envs[-1],bound1[i],tfd_bra[i,:],tfd_ket[i,:],bound2[i],XMAX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['l','L',0]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L1) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(upto):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_L(envs[-1],bound1[j],tfd_bra[:,j],tfd_ket[:,j],bound2[j],XMAX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['r','R',3]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L1) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(L2-1,upto-1,-1):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_R(envs[-1],bound1[j],tfd_bra[:,j],tfd_ket[:,j],bound2[j],XMAX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]


    if get_errs:   return envs, errs
    else:          return envs



#############################################
####  apply lambdas to boundary mpos    #####
#############################################


def apply_lam_to_boundary_mpo(b_mpo,bonds_u,bonds_d,op='sqrt_inv'):

    return ENV_GL.apply_lam_to_boundary_mpo(b_mpo,bonds_u,bonds_d,op)


#############################################
#####     ovlps from boundaries     #########
#############################################

def ovlp_from_bound(bound,to_squeeze=True):
    ''' obtain ovlp from bound which is from contracting entire peps by row/col
        should be mpo with physical bonds (1,1)
    '''
    return ENV_GL.ovlp_from_bound(bound,to_squeeze)


def contract_2_bounds(bound1, bound2, bonds_u, bonds_d):
    ''' eg. bL -- bR, bI -- bO '''

    return ENV_GL.contract_2_bounds(bound1,bound2)


#############################################
#####         contract full env     #########
#############################################

## can also be done from PEPX class, dotting bra+ket and calling PEP0_env contract fct ##

## alternative method (following boundary method more closely ##

def get_norm(tldm, side='I',XMAX=100,get_err=False):

    return get_ovlp(np.conj(tldm), tldm, side, XMAX, get_err)


def get_ovlp(bra, ket, side='I',XMAX=100,get_err=False):
    ''' contract virtual bonds of pep0 (no physical bonds) '''
   
    L1, L2 = tfd[0].shape

    if   side in ['i','I',1]:   upto = L1
    elif side in ['l','L',0]:   upto = L2
    else:                       upto = 0

    boundaries = get_boundaries(bra,ket,side,upto,XMAX=XMAX,get_err=get_err)
    
    if not get_err:   
        bound = boundaries[-1]

        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        return ovlp
    else:
        bounds, errs = boundaries
        bound = bounds[-1]
       
        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        return ovlp, errs



#############################################################
####          embedded methods (on the fly env)          ####
#############################################################

def embed_sites_norm(sub_tldm,envs_list,side='L',XMAX=100,get_errs=False):
    ''' get norm of sub_pepx embedded in env '''

    # out =  embed_sites_ovlp( np.conj(PEPX.transposeUD(sub_pepx)), sub_pepx, envs_list, side, XMAX, get_errs)
    out =  embed_sites_ovlp( np.conj(sub_tldm), sub_tldm, envs_list, side, XMAX, get_errs)
    if get_errs:
        norm2, errs = out
        return np.sqrt(norm2), errs
    else:
        return np.sqrt(out)


def embed_sites_ovlp(sub_bra, sub_ket, envs_list, side='L',XMAX=100,get_errs=False):
    ''' get ovlp of sub_pepx_u, sub_pepx_d embedded in env 
        note:  input bra is not already conj transposed
    '''

    L1, L2 = sub_bra.shape
    bL, bI, bO, bR = envs_list

    cum_err = 0
    if   side in ['l','L',0]:
        sb = bL
        for j in range(L2):
            subbound, err = get_next_subboundary_L(sb,bI[j],sub_bra[:,j],sub_ket[:,j],bO[j],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bR)

    elif side in ['r','R',3]:
        sb = bR
        for j in range(L2)[::-1]:
            subbound, err = get_next_subboundary_R(sb,bI[j],sub_bra[:,j],sub_ket[:,j],bO[j],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bL)

    elif side in ['i','I',1]:
        sb = bI
        for i in range(L1):
            subbound, err = get_next_subboundary_I(sb,bL[i],sub_bra[i,:],sub_ket[i,:],bR[i],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bO)

    elif side in ['o','O',2]:
        sb = bO
        for i in range(L1)[::-1]:
            subbound, err = get_next_subboundary_O(sb,bL[i],sub_bra[i,:],sub_ket[i,:],bR[i],XMAX=XMAX)
            sb = subbound
            cum_err += err
        ovlp = contract_2_bounds(sb,bI)


    if get_errs:    return ovlp, cum_err
    else:           return ovlp
 


def build_env(envs,ensure_pos=False):
    # env MPOs already do NOT have lambdas on open bonds
    return ENV.build_env(envs,ensure_pos)


