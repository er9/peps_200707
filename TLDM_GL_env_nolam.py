import numpy as np
import time
import tens_fcts as tf

import MPX
import PEPX
import PEPX_GL
import PEPS_GL_env_nolam as ENV_GL


########################################################
####               get boundary MPOs                ####

''' get boundaries by contracting rows/cols of pepx 
    obtain list of boundary MPOs/MPSs,
    up to row/col "upto"
    returns err from compression of MPO/MPS if desired

    note:  bonds/lambdas for legs sticking out are not included
'''
########################################################

#### grouping of gamma/lambda makes it look right canon? can we make code s.t. canonicalization step
#### is not necessary?

def get_next_boundary_O(tldm_u_row,tldm_d_row,boundary,XMAX=100):
    '''  get outside boundary (facing in)
         note:  peps_x_row is 2-D, but a 1 x L2 pepx_GL
    '''
    
    peps_u_row = PEPX_GL.flatten(tldm_u_row)
    peps_d_row = PEPX_GL.flatten(tldm_d_row)
    return  ENV_GL.get_next_boundary_O(peps_u_row,peps_d_row,boundary,XMAX)

    # L2 = len(tldm_u_row)
    # boundary_mpo = boundary.copy()

    # for j in range(L2):
    #     x_lam = [0,1]
    #     ptens_u = PEPX_GL.get_site(tldm_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_row,j,no_lam=x_lam)

    #     # tens2 = np.einsum('liord,LIORd,xoOy->xlLiIyrR',ptens_u,ptens_d,boundary_mpo[j])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens2 = np.einsum('LIORda,xoOy->xLIoyRda',ptens_d,boundary_mpo[j])
    #         tens2 = np.einsum('liorda,xLIoyRda->xlLiIyrR',ptens_u,tens2)
    #     else:
    #         tens2 = np.einsum('liorda,xoOy->xliOyrda',ptens_u,boundary_mpo[j])
    #         tens2 = np.einsum('LIORda,xliOyrda->xlLiIyrR',ptens_d,tens2)
    #     boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

 
    # # err = np.nan
    # boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    # boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    # if np.max(err) > 1.0e-1:
    #     print [np.linalg.norm(m) for m in boundary_mpo]
    #     print err
    #     raise RuntimeWarning('bound O large compression error, XMAX'+str(XMAX))

    # return boundary_mpo, err


def get_next_boundary_I(tldm_u_row,tldm_d_row,boundary,XMAX=100):
    ''' get inside boundary (facing out)
        note:  peps_x_row is 1-D
    '''

    peps_u_row = PEPX_GL.flatten(tldm_u_row)
    peps_d_row = PEPX_GL.flatten(tldm_d_row)
    return  ENV_GL.get_next_boundary_I(peps_u_row,peps_d_row,boundary,XMAX)

    # L2 = len(tldm_u_row)
    # boundary_mpo = boundary.copy()

    # for j in range(L2):

    #     x_lam = [2,3]
    #     ptens_u = PEPX_GL.get_site(tldm_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_row,j,no_lam=x_lam)

    #     # tens2 = np.einsum('xiIy,liord,LIORd->xlLoOyrR',boundary_mpo[j],peps_u_row[j],peps_d_row[j])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens2 = np.einsum('xiIy,LIORda->xLiOyRda',boundary_mpo[j],ptens_d)
    #         tens2 = np.einsum('xLiOyRda,liorda->xlLoOyrR',tens2,ptens_u)
    #     else:
    #         tens2 = np.einsum('xiIy,liorda->xlIoyrda',boundary_mpo[j],ptens_u)
    #         tens2 = np.einsum('xlIoyrda,LIORda->xlLoOyrR',tens2,ptens_d)
    #     boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

    # # err = np.nan
    # boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    # boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,0)

    # if np.max(err) > 1.0e-1:
    #     print [np.linalg.norm(m) for m in boundary_mpo]
    #     print err
    #     raise RuntimeWarning('bound I large compression error, XMAX'+str(XMAX))

    # return boundary_mpo, err


def get_next_boundary_L(tldm_u_col,tldm_d_col,boundary,XMAX=100):
    ''' get left boundary (facing right)
        note:  peps_x_col is 1-D
    '''

    peps_u_col = PEPX_GL.flatten(tldm_u_col)
    peps_d_col = PEPX_GL.flatten(tldm_d_col)
    return  ENV_GL.get_next_boundary_L(peps_u_col,peps_d_col,boundary,XMAX)

    # L1 = len(tldm_u_col)
    # boundary_mpo = boundary.copy()

    # for i in range(L1):

    #     x_lam = [1,3]
    #     ptens_u = PEPX_GL.get_site(tldm_u_col,i,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_col,i,no_lam=x_lam)

    #     # tens2 = np.einsum('xlLy,liord,LIORd->xiIrRyoO',boundary_mpo[i],peps_u_col[i],peps_d_col[i])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens2 = np.einsum('xlLy,LIORda->xlIORyda',boundary_mpo[i],ptens_d)
    #         tens2 = np.einsum('xlIORyda,liorda->xiIrRyoO',tens2,ptens_u)
    #     else:
    #         tens2 = np.einsum('xlLy,liorda->xLioryda',boundary_mpo[i],ptens_u)
    #         tens2 = np.einsum('xLioryda,LIORda->xiIrRyoO',tens2,ptens_d)
    #     boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')
    # 
    # # err = np.nan
    # boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    # boundary_mpo, err  = MPX.compress_reg(boundary_mpo,XMAX,0)

    # if np.max(err) > 1.0e-1:
    #     print [np.linalg.norm(m) for m in boundary_mpo]
    #     print err
    #     raise RuntimeWarning('bound L large compression error, XMAX'+str(XMAX))

    # return boundary_mpo, err


def get_next_boundary_R(tldm_u_col,tldm_d_col,boundary,XMAX=100):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''

    peps_u_col = PEPX_GL.flatten(tldm_u_col)
    peps_d_col = PEPX_GL.flatten(tldm_d_col)
    return  ENV_GL.get_next_boundary_R(peps_u_col,peps_d_col,boundary,XMAX)

    # L1 = len(tldm_u_col)
    # boundary_mpo = boundary.copy()

    # for i in range(L1):

    #     x_lam = [0,2]
    #     ptens_u = PEPX_GL.get_site(tldm_u_col,i,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_col,i,no_lam=x_lam)

    #     # tens2 = np.einsum('liord,LIORd,xrRy->xiIlLyoO',peps_u_col[i],peps_d_col[i],boundary_mpo[i])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens2 = np.einsum('LIORda,xrRy->xLIOryda',ptens_d,boundary_mpo[i])
    #         tens2 = np.einsum('liorda,xLIOryda->xiIlLyoO',ptens_u,tens2)
    #     else:
    #         tens2 = np.einsum('liorda,xrRy->xlioRyda',ptens_u,boundary_mpo[i])
    #         tens2 = np.einsum('LIORda,xlioRyda->xiIlLyoO',ptens_d,tens2)
    #     boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')

    # # err = np.nan
    # boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    # boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    # if np.max(err) > 1.0e-1:
    #     print [np.linalg.norm(m) for m in boundary_mpo]
    #     print err
    #     raise RuntimeWarning('bound R large compression error, XMAX'+str(XMAX))

    # return boundary_mpo, err


#####################
#### full method ####
#####################

def get_boundaries(tldm_bra,tldm_ket,side,upto,init_bound=None,XMAX=100,get_err=False):
    ''' get_boundaries for pep0 (no dangling physical bonds, e.g. traced out) 
    '''

    L1, L2 = tldm_ket.shape

    if side in ['o','O',2]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]

        for i in range(L1-1,upto-1,-1):      # building envs from outside to inside
            boundary_mpo, err = get_next_boundary_O(tldm_bra[i,:],tldm_ket[i,:],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]
            
    elif side in ['i','I',1]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
     
        for i in range(upto):              # building envs from inside to outside
            # print 'tldm bound i',[np.linalg.norm(m) for m in envs[-1]]
            # print [np.linalg.norm(m) for m in tldm_bra[i,:]]
            # print [np.linalg.norm(m) for m in tldm_ket[i,:]]

            boundary_mpo, err = get_next_boundary_I(tldm_bra[i,:],tldm_ket[i,:],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
    
    elif side in ['l','L',0]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L1) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(upto):              # building envs from left to right
            boundary_mpo, err = get_next_boundary_L(tldm_bra[:,j],tldm_ket[:,j],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
    
       
    elif side in ['r','R',3]:

        if init_bound is None:        envs = [ MPX.ones([(1,1)]*L1) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(L2-1,upto-1,-1):      # building envs from right to left
            boundary_mpo, err = get_next_boundary_R(tldm_bra[:,j],tldm_ket[:,j],envs[-1],XMAX)
            envs.append( boundary_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]
    
    else:
        raise ValueError('get boundaries, provide valid direction, not '+str(direction))
    
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

def get_next_subboundary_O(env_mpo,boundL_tens,tldm_u_row,tldm_d_row,boundR_tens,XMAX=100,c_tol=1.0e-8):

    peps_u_row = PEPX_GL.flatten(tldm_u_row)
    peps_d_row = PEPX_GL.flatten(tldm_d_row)
    return  ENV_GL.get_next_subboundary_O(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX,c_tol)

    # L2 = len(tldm_u_row)
    # e_mpo = env_mpo.copy()

    # for j in range(L2):

    #     if j == 0:     x_lam = [1] 
    #     else:          x_lam = [0,1]

    #     ptens_u = PEPX_GL.get_site(tldm_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_row,j,no_lam=x_lam)

    #     # tens = np.einsum('liord,LIORd,xoOy->lLxiIrRy', peps_u_row[j],peps_d_row[j],e_mpo[j])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens = np.einsum('LIORda,xoOy->LxIoRyda',ptens_d,e_mpo[j])
    #         tens = np.einsum('liorda,LxIoRyda->lLxiIrRy',ptens_u,tens)
    #     else:
    #         tens = np.einsum('liorda,xoOy->lxiOryda',ptens_u,e_mpo[j])
    #         tens = np.einsum('LIORda,lxiOryda->lLxiIrRy',ptens_d,tens)
    #     e_mpo[j] = tens  # tf.reshape(tens,'iii,i,i,iii')

    # # contract envs with boundary 1, boundary 2
    # e_mpo[0]  = np.einsum('wlLx,lLxiIrRy->wiIrRy',boundL_tens,e_mpo[0])   # i(rRo) -- (lLx)oO(rRy)
    # e_mpo[-1] = np.einsum('...rRy,zrRy->...z',e_mpo[-1],boundR_tens)      # (lLx)oO(rRy) -- i(lLo)

    # if L2 > 1:
    #     e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
    #     e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
    #     for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')
    # 
    # # err = np.nan
    # e_mpo  = MPX.canonicalize(e_mpo,1)
    # e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0)

    # if len(err) > 0:
    #     if np.max(err) > 1.0e-1:
    #         print [np.linalg.norm(m) for m in e_mpo]
    #         print err
    #         raise RuntimeWarning('subO large compression error, XMAX '+str(XMAX))

    # return e_mpo, err


def get_next_subboundary_I(env_mpo,boundL_tens,tldm_u_row,tldm_d_row,boundR_tens,XMAX=100,c_tol=1.0e-8):

    peps_u_row = PEPX_GL.flatten(tldm_u_row)
    peps_d_row = PEPX_GL.flatten(tldm_d_row)
    return  ENV_GL.get_next_subboundary_I(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX,c_tol)

    # L2 = len(tldm_u_row)
    # e_mpo = env_mpo.copy()
    # 
    # for j in range(L2):

    #     if j == L2-1:     x_lam = [2]
    #     else:             x_lam = [2,3]

    #     ptens_u = PEPX_GL.get_site(tldm_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_row,j,no_lam=x_lam)

    #     # tens = np.einsum('xiIy,liord,LIORd->xlLoOyrR', e_mpo[j],peps_u_row[j],peps_d_row[j])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens = np.einsum('xiIy,LIORda->xLiOyRda',e_mpo[j],ptens_d)
    #         tens = np.einsum('xLiOyRda,liorda->xlLoOyrR',tens,ptens_u)
    #     else:
    #         tens = np.einsum('xiIy,liorda->xlIoyrda',e_mpo[j],ptens_u)
    #         tens = np.einsum('xlIoyrda,LIORda->xlLoOyrR',tens,ptens_d)
    #     e_mpo[j] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # # contract envs with boundary 1, boundary 2
    # e_mpo[0]  = np.einsum('xlLw,xlLoOyrR->woOyrR',boundL_tens,e_mpo[0])   # (irR)o -- (xlL)iI(yrR)
    # e_mpo[-1] = np.einsum('...yrR,yrRz->...z',e_mpo[-1],boundR_tens)      # (xlL)iI(yrR) -- (ilL)o

    # if L2 > 1:
    #     e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
    #     e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
    #     for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')
    # 
    # # err = np.nan
    # e_mpo  = MPX.canonicalize(e_mpo,0)
    # e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1)

    # if len(err) > 0:
    #     if np.max(err) > 1.0e-1:
    #         print [np.linalg.norm(m) for m in e_mpo]
    #         print err
    #         raise RuntimeWarning('subI large compression error, XMAX '+str(XMAX))

    # return e_mpo, err


def get_next_subboundary_L(env_mpo,boundI_tens,tldm_u_col,tldm_d_col,boundO_tens,XMAX=100,c_tol=1.0e-8):

    peps_u_col = PEPX_GL.flatten(tldm_u_col)
    peps_d_col = PEPX_GL.flatten(tldm_d_col)
    return  ENV_GL.get_next_subboundary_L(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX,c_tol)

    # L1 = len(tldm_u_col)

    # e_mpo = env_mpo.copy()
    # 
    # for i in range(L1):

    #     if i == 0:   x_lam = [3]
    #     else:        x_lam = [1,3]   # looks right canonical

    #     ptens_u = PEPX_GL.get_site(tldm_u_col,i,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_col,i,no_lam=x_lam)

    #     # tens = np.einsum('xlLy,liord,LIORd->xiIrRyoO', e_mpo[i],peps_u_col[i],peps_d_col[i])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens = np.einsum('xlLy,LIORda->xIlRyOda',e_mpo[i],ptens_d)
    #         tens = np.einsum('xIlRyOda,liorda->xiIrRyoO',tens,ptens_u)
    #     else:
    #         tens = np.einsum('xlLy,liorda->xiLryoda',e_mpo[i],ptens_u)
    #         tens = np.einsum('xiLryoda,LIORda->xiIrRyoO',tens,ptens_d)
    #     e_mpo[i] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # # contract envs with boundary 1, boundary 2
    # # print 'subL', boundI_tens.shape, e_mpo[0].shape
    # e_mpo[0]  = np.einsum('xiIw,xiIrRyoO->wrRyoO',boundI_tens,e_mpo[0])    # l(oOr) -- (xiI)rR(yoO)
    # e_mpo[-1] = np.einsum('...yoO,yoOz->...z',e_mpo[-1],boundO_tens)       # (xiI)rR(yoO) -- (liI)r


    # if L1 > 1:
    #     e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
    #     e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
    #     for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')

    # # err = np.nan
    # e_mpo  = MPX.canonicalize(e_mpo,0)
    # e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1)

    # if len(err) > 0:
    #     if np.max(err) > 1.0e-1:
    #         print [np.linalg.norm(m) for m in e_mpo]
    #         print [np.linalg.norm(m) for m in env_mpo], np.linalg.norm(boundI_tens), np.linalg.norm(boundO_tens)
    #         print [np.linalg.norm(m) for m in tldm_u_col]
    #         print [np.linalg.norm(m) for m in tldm_d_col]
    #         print err
    #         raise RuntimeWarning('subL large compression error, XMAX '+str(XMAX))

    # return e_mpo, err


def get_next_subboundary_R(env_mpo,boundI_tens,tldm_u_col,tldm_d_col,boundO_tens,XMAX=100,c_tol=1.0e-8):

    peps_u_col = PEPX_GL.flatten(tldm_u_col)
    peps_d_col = PEPX_GL.flatten(tldm_d_col)
    return  ENV_GL.get_next_subboundary_R(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX,c_tol)

    # L1 = len(tldm_u_col)
    # e_mpo = env_mpo.copy()


    # for i in range(L1):

    #     if i == L1-1:      x_lam = [0]
    #     else:              x_lam = [0,2]

    #     ptens_u = PEPX_GL.get_site(tldm_u_col,i,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(tldm_d_col,i,no_lam=x_lam)

    #     # tens = np.einsum('liord,LIORd,xrRy->iIxlLoOy', peps_u_col[i],peps_d_col[i],e_mpo[i])
    #     if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
    #         tens = np.einsum('LIORda,xrRy->IxLrOyda',ptens_d,e_mpo[i])
    #         tens = np.einsum('liorda,IxLrOyda->iIxlLoOy',ptens_u,tens)
    #     else:
    #         tens = np.einsum('liorda,xrRy->ixlRoyda',ptens_u,e_mpo[i])
    #         tens = np.einsum('LIORda,ixlRoyda->iIxlLoOy',ptens_d,tens)
    #     e_mpo[i] = tens    # tf.reshape(tens,'iii,i,i,iii')

    # # contract envs with boundary 1, boundary 2
    # e_mpo[0]  = np.einsum('wiIx,iIxlLoOy->wlLoOy',boundI_tens,e_mpo[0])    # l(oOr) -- (iIx)lL(oOy)
    # e_mpo[-1] = np.einsum('...oOy,zoOy->...z',e_mpo[-1],boundO_tens)       # (iIx)lL(oOy) -- l(iIr)

    # if L1 > 1:    # if L1 == 1 don't need to do any reshaping
    #     e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
    #     e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
    #     for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')
    # 
    # # err = np.nan
    # e_mpo  = MPX.canonicalize(e_mpo,1)
    # e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0)

    # if len(err) > 0:
    #     if np.max(err) > 1.0e-1:
    #         print [np.linalg.norm(m) for m in e_mpo]
    #         print [np.linalg.norm(m) for m in env_mpo], np.linalg.norm(boundI_tens), np.linalg.norm(boundO_tens)
    #         print [np.linalg.norm(m) for m in tldm_u_col]
    #         print [np.linalg.norm(m) for m in tldm_d_col]
    #         print err
    #         raise RuntimeWarning('subR large compression error, XMAX '+str(XMAX))

    # return e_mpo, err


#####################
#### full method ####
#####################

def get_subboundaries(bound1,bound2,tldm_bra,tldm_ket,side,upto,init_sb=None,XMAX=100,c_tol=1.0e-8,get_errs=False):
    
    L1, L2 = tldm_ket.shape  

    if side in ['o','O',2]:    # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(L1-1,upto-1,-1):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_O(envs[-1],bound1[i],tldm_bra[i,:],tldm_ket[i,:],bound2[i],XMAX,c_tol)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]

    elif side in ['i','I',1]:   # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(upto):   # get envs up to row upto
            e_mpo, err = get_next_subboundary_I(envs[-1],bound1[i],tldm_bra[i,:],tldm_ket[i,:],bound2[i],XMAX,c_tol)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['l','L',0]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L1) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(upto):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_L(envs[-1],bound1[j],tldm_bra[:,j],tldm_ket[:,j],bound2[j],XMAX,c_tol)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['r','R',3]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,1)]*L1) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(L2-1,upto-1,-1):   # get envs up to col upto
            e_mpo, err = get_next_subboundary_R(envs[-1],bound1[j],tldm_bra[:,j],tldm_ket[:,j],bound2[j],XMAX,c_tol)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]


    if get_errs:   return envs, errs
    else:          return envs



#############################################
####  apply lambdas to boundary mpos    #####
#############################################


def apply_lam_to_boundary_mpo(b_mpo,bonds_u,bonds_d,op='none'):

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

    return ENV_GL.contract_2_bounds(bound1,bound2,bonds_u,bonds_d)
 

#############################################
#####         contract full env     #########
#############################################

def get_norm(tldm, side='I',XMAX=100,get_err=False):

    norm2 = get_ovlp(np.conj(tldm), tldm, side, XMAX, get_err=get_err)
    return np.sqrt(np.squeeze(norm2))


def get_ovlp(bra, ket, side='I',XMAX=100,get_err=False):
    ''' contract virtual bonds of pep0 (no physical bonds) '''
   
    L1, L2 = ket.shape

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

        # ind = 0
        # for b in bounds:
        #     print 'in get ovlp', len(b), b[0].shape, ovlp
        #     if len(b) == 1:
        #         sqdim = b[0].shape[1]
        #         const = 1./ovlp**(ind*1./max(L1,L2))
        #         if not np.allclose( b[0].reshape(sqdim,sqdim)*const, np.eye(sqdim)):
        #             print 'get ovlp not canonical'
        #             print b[0].reshape(sqdim,sqdim)*const
        #         else:
        #             print 'get ovlp ok canonical'

        #     ind += 1
    

        return ovlp, errs


def get_sub_ovlp(sub_tldm_u, sub_tldm_d, bounds, side=None, XMAX=100, get_err=False):

    return ENV_GL.get_sub_ovlp(sub_tldm_u,sub_tldm_d,bounds,side,XMAX,get_err,sb_fn=get_subboundaries)

    # L1, L2 = sub_pepx_u.shape

    # if side is None:
    #     if L1 >= L2:  side = 'I'
    #     else:         side = 'L'

    # if   side in ['i','I',1]:   upto = L1
    # elif side in ['l','L',0]:   upto = L2
    # else:                       upto = 0

    # bL, bI, bO, bR = bounds
    # if not get_err:
    #     if side   in ['i','I',1]:
    #         bounds = get_subboundaries(bL,bR,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bI,XMAX=XMAX,get_errs=False)
    #         ovlp  = contract_2_bounds(bounds[upto], bO)
    #     elif side in ['o','O',2]:
    #         bounds = get_subboundaries(bL,bR,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bO,XMAX=XMAX,get_errs=False)
    #         ovlp  = contract_2_bounds(bounds[upto], bI)
    #     elif side in ['l','L',0]:
    #         bounds = get_subboundaries(bI,bO,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bL,XMAX=XMAX,get_errs=False)
    #         ovlp  = contract_2_bounds(bounds[upto], bL)
    #     elif side in ['r','R',3]:
    #         bounds = get_subboundaries(bI,bO,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bR,XMAX=XMAX,get_errs=False)
    #         ovlp  = contract_2_bounds(bounds[upto], bR)

    #     return ovlp

    # else:
    #     if side   in ['i','I',1]:
    #         bounds,errs=get_subboundaries(bL,bR,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bI,XMAX=XMAX,get_errs=True)
    #         ovlp  = contract_2_bounds(bounds[upto], bO)
    #     elif side in ['o','O',2]:
    #         bounds,errs=get_subboundaries(bL,bR,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bO,XMAX=XMAX,get_errs=True)
    #         ovlp  = contract_2_bounds(bounds[upto], bI)
    #     elif side in ['l','L',0]:
    #         bounds,errs=get_subboundaries(bI,bO,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bL,XMAX=XMAX,get_errs=True)
    #         ovlp  = contract_2_bounds(bounds[upto], bL)
    #     elif side in ['r','R',3]:
    #         bounds,errs=get_subboundaries(bI,bO,sub_tldm_u,sub_tldm_d,side,upto,init_sb=bR,XMAX=XMAX,get_errs=True)
    #         ovlp  = contract_2_bounds(bounds[upto], bR)

    #     return ovlp, errs


#############################################################
####          embedded methods (on the fly env)          ####
#############################################################

def embed_sites_xog_2x2_bond(sub_tldm_u, sub_tldm_d, env_list, g_opt, idx_list, idx_conns):

    def get_bL10(bL1,tens10_u,tens10_d,bO0):
        # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_tldm_u[1,0],sub_tldm_d[1,0])
        bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        bL10 = np.einsum('wlLoOy,liorda->wLiOryda',bL10,tens10_u)
        bL10 = np.einsum('wLiOryda,LIORda->wiIrRy',bL10,tens10_d)
        return bL10

    def get_bL11(bO1,tens11_u,tens11_d,bR1):
        # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])
        bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        bL11 = np.einsum('xoOrRz,liorda->xliORzda',bL11,tens11_u)
        bL11 = np.einsum('xliORzda,LIORda->xlLiIz',bL11,tens11_d)
        return bL11

    def get_bL01(bI1,tens01_u,tens01_d,bR0):
        # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])
        bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        bL01 = np.einsum('xiIrRz,liorda->xlIoRzda',bL01,tens01_u)
        bL01 = np.einsum('xlIoRzda,LIORda->xlLoOz',bL01,tens01_d)
        return bL01

    def get_bL00(bL0,tens00_u,tens00_d,bI0):
        # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])
        bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        bL00 = np.einsum('xlLiIz,liorda->xLIorzda',bL00,tens00_u)
        bL00 = np.einsum('xLIorzda,LIORda->xoOrRz',bL00,tens00_d)
        return bL00


    # with g_opt (contract at position 2)
    def get_bL10_g(bL1,tens10_u,tens10_d,bO0,g_opt):
        # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])
        bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        bL10 = np.einsum('wlLoOy,liorda->wLiOryda',bL10,tens10_u)
        bL10 = np.einsum('wLiOryda,LIORdA->wiIrRyaA',bL10,tens10_d)
        bL10 = np.einsum('wiIrRybB,ABab->wiIrRyaA',bL10,g_opt)
        return bL10

    def get_bL11_g(bO1,tens11_u,tens11_d,bR1,g_opt):
        # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])
        bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        bL11 = np.einsum('xoOrRz,liorda->xliORzda',bL11,tens11_u)
        bL11 = np.einsum('xliORzda,LIORdA->xlLiIzaA',bL11,tens11_d)
        bL11 = np.einsum('xlLiIzbB,ABab->xlLiIzaA',bL11,g_opt)
        return bL11

    def get_bL01_g(bI1,tens01_u,tens01_d,bR0,g_opt):
        # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])
        bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        bL01 = np.einsum('xiIrRz,liorda->xlIoRzda',bL01,tens01_u)
        bL01 = np.einsum('xlIoRzda,LIORdA->xlLoOzaA',bL01,tens01_d)
        bL01 = np.einsum('xlLoOzbB,ABab->xlLoOzaA',bL01,g_opt)
        return bL01

    def get_bL00_g(bL0,tens00_u,tens00_d,bI0,g_opt):
        # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])
        bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        bL00 = np.einsum('xlLiIz,liorda->xLIorzda',bL00,tens00_u)
        bL00 = np.einsum('xLIorzda,LIORdA->xoOrRzaA',bL00,tens00_d)
        bL00 = np.einsum('xoOrRzbB,ABab->xoOrRzaA',bL00,g_opt)
        return bL00


    ### order contractions assuming 'rol' mpo connectivity ###
    ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
    ###                               10 < 00 ~ 11 < 01 for 3-body operator
    if x_idx == (0,0):

        su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1,3])
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1,3])
        su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[])
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[])
        su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0,2])
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0,2])

        ## 00 corner 
        # su00 = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[])

        if iso_leg in ['r','R',3]:

            bL10 = get_bL10(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])
            bL01 = get_bL01_g(bI[1],su01,sd01,bR[0],g_opt)

            bLs = np.einsum('wiIrRx,xrRoOy->wiIoOy',bL10,bL11)
            bLs = np.einsum('wiIoOy,zlLoOyaA->wiIlLzaA',bLs,bL01)

            if qr_reduce:
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                bL00 = np.einsum('xlLiIz,lioq->xLIoqz',bL00,qu_idx)
                bL00 = np.einsum('xLIoqz,LIORdA->xoOqRzdA',bL00,sd00)
                env_out = np.einsum('xoOqRzdA,xoOrRzaA->qrda',bL00,bLs)
            else:
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                bL00 = np.einsum('xlLiIz,LIORdA->xliORdzA',bL00,sd00)
                env_out = np.einsum('xliORdzA,xoOrRzaA->liorda',bL00,bLs)
     
        elif iso_leg in ['o','O',2]:

            bL10 = get_bL10_g(bL[1],su10,sd10,bO[0],g_opt)
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])

            bLs = np.einsum('zlLoOy,xrRoOy->xlLrRz',bL01,bL11)
            bLs = np.einsum('wiIrRxaA,xlLrRz->wiIlLzaA',bL10,bLs)

            if qr_reduce:
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                bL00 = np.einsum('xlLiIz,lirq->xLIqrz',bL00,qu_idx)
                bL00 = np.einsum('xLIqrz,LIORdA->xqOrRzdA',bL00,sd00)
                env_out = np.einsum('xqOrRzdA,xoOrRzaA->qoda',bL00,bLs)
            else:
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                bL00 = np.einsum('xlLiIz,LIORdA->xliORdzA',bL00,sd00)
                env_out = np.einsum('xliORdzA,xoOrRzaA->liorda',bL00,bLs)


    elif x_idx == (0,1):

        su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[])
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[])
        su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[0,1])
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[0,1])
        su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2,3])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2,3])

        ## 01 corner
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[])


        if iso_leg in ['o','O',2]:   # update (0,1)-(1,1) bond
            bL10 = get_bL10(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11_g(bO[1],su11,sd11,bR[1],g_opt)
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])

            bLs = np.einsum('xoOrRw,xoOlLy->wrRlLy',bL00,bL10)
            bLs = np.einsum('wrRlLy,ylLiIzaA->wrRiIzaA',bLs ,bL11)

            if qr_reduce:
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                bL01 = np.einsum('xiIrRz,lirq->xlIqRz',bL01,qu_idx)
                bL01 = np.einsum('xlIqRz,LIORdA->xlLqOzdA',bL01,sd01)
                env_out = np.einsum('xlLqOzdA,xlLoOzaA->qoda',bL01,bLs)
            else:
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                bL01 = np.einsum('xiIrRz,LIORdA->xLiOrdAz',bL01,sd01)
                env_out = np.einsum('xlLoOzaA,xLiOrdAz->liorda',bLs,bL01)

        elif iso_leg in ['l','L',0]:  # update (0,1)-(0,0) bond
            bL10 = get_bL10(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])
            bL00 = get_bL00_g(bL[0],su00,sd00,bI[0],g_opt)

            bLs = np.einsum('xoOlLy,ylLiIz->xoOiIz',bL10,bL11)
            bLs = np.einsum('xoOrRwaA,xoOiIz->wrRiIzaA',bL00,bLs)

            if qr_reduce:
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                bL01 = np.einsum('xiIrRz,iorq->xqIoRz',bL01,qu_idx)
                bL01 = np.einsum('xqIoRz,LIORdA->xqLoOzdA',bL01,sd01)
                env_out = np.einsum('xqLoOzdA,xlLoOzaA->qlda',bL01,bLs)
            else:
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                bL01 = np.einsum('xiIrRz,LIORdA->xLiOrdAz',bL01,sd01)
                env_out = np.einsum('wlLoOzaA,wLiOrdAz->liorda',bLs,bL01)
        
    elif x_idx == (1,0):

        su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[])
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[])
        su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[0,1])
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[0,1])
        su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2,3])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2,3])

        ## 10 corner
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[])

        if iso_leg in ['r','R',3]:     # update (1,0) - (1,1) bond
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])
            bL11 = get_bL11_g(bO[1],su11,sd11,bR[1],g_opt)

            bLs = np.einsum('woOrRx,xrRiIy->woOiIy',bL00,bL01)
            bLs = np.einsum('woOiIy,zlLiIyaA->woOlLzaA',bLs, bL11)

            if qr_reduce:
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                bL10 = np.einsum('xlLoOz,lioq->xLiOqz',bL10,qu_idx)
                bL10 = np.einsum('xLiOqz,LIORdA->xiIqRzdA',bL10,sd10)
                env_out = np.einsum('xiIqRzdA,xiIrRzaA->qrda',bL10,bLs)
            else:
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                bL10 = np.einsum('xlLoOz,LIORdA->xlIoRdAz',bL10,sd10)
                env_out = np.einsum('xlIoRdAz,xiIrRzaA->liorda',bL10,bLs)

        elif iso_leg in ['i','I',1]:   # update (1,0) - (0,0) bond 
            bL00 = get_bL00_g(bL[0],su00,sd00,bI[0],g_opt)
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])

            bLs = np.einsum('xrRiIy,zlLiIy->xrRlLz',bL01,bL11)
            bLs = np.einsum('woOrRxaA,xrRlLz->woOlLzaA',bL00,bLs)

            if qr_reduce:
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                bL10 = np.einsum('xlLoOz,lorq->xLqOrz',bL10,qu_idx)
                bL10 = np.einsum('xLqOrz,LIORdA->xqIrRzdA',bL10,sd10)
                env_out = np.einsum('xqIrRzdA,xiIrRzaA->qida',bL10,bLs)
            else:
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                bL10 = np.einsum('xlLoOz,LIORdA->xlIoRdAz',bL10,sd10)
                env_out = np.einsum('xlIoRdAz,xiIrRzaA->liorda',bL10,bLs)


    elif x_idx == (1,1):

        su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1,3])
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1,3])
        su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[])
        su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0,2])
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0,2])

        ## 11 corner
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[])

        if iso_leg in ['l','L',0]:      # update (1,1) - (1,0)
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])
            bL10 = get_bL10_g(bL[1],su10,sd10,bO[0],g_opt)

            bLs = np.einsum('xiIlLy,ylLoOz->xiIoOz',bL00,bL01)
            bLs = np.einsum('xiIrRwaA,xiIoOz->wrRoOzaA',bL10,bLs)

            if qr_reduce:
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                bL11 = np.einsum('xoOrRz,iorq->xqiORz',bL11,qu_idx)
                bL11 = np.einsum('xqiORz,LIORdA->xqLiIzdA',bL11,sd11)
                env_out = np.einsum('xqLiIzdA,xlLiIzaA->qlda',bL11,bLs)
            else:
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                bL11 = np.einsum('xoOrRz,LIORdA->xLIordAz',bL11,sd11)
                env_out = np.einsum('xlLiIzaA,xLIordAz->liorda',bLs,bL11)

        elif iso_leg in ['i','I',1]:    # update (1,1) - (0,1)
            bL01 = get_bL01_g(bI[1],su01,sd01,bR[0],g_opt)
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])
            bL10 = get_bL10(bL[1],su10,sd10,bO[0])

            bLs = np.einsum('xiIrRw,xiIlLy->wrRlLy',bL10,bL00)
            bLs = np.einsum('wrRlLy,ylLoOzaA->wrRoOzaA',bLs, bL01)

            if qr_reduce:
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                bL11 = np.einsum('xoOrRz,lorq->xlqORz',bL11,qu_idx)
                bL11 = np.einsum('xlqORz,LIORdA->xlLqIzdA',bL11,sd11)
                env_out = np.einsum('xlLqIzdA,xlLiIzaA->qida',bL11,bLs)
            else:
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                bL11 = np.einsum('xoOrRz,LIORdA->xLIordAz',bL11,sd11)
                env_out = np.einsum('xlLiIzaA,xLIordAz->liorda',bLs,bL11)

    else:  raise (IndexError)


def embed_sites_oog_2x2_bond(sub_tldm_u, sub_tldm_d, env_list, x_idx, iso_leg):

    def get_bL10(bL1,tens10_u,tens10_d,bO0):
        # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])
        bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        bL10 = np.einsum('wlLoOy,liorda->wLiOryda',bL10,tens10_u)
        bL10 = np.einsum('wLiOrydA,LIORda->wiIrRy',bL10,tens10_d)
        return bL10
 
    def get_bL11(bO1,tens11_u,tens11_d,bR1):
        # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])
        bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        bL11 = np.einsum('xoOrRz,liorda->xliORzda',bL11,tens11_u)
        bL11 = np.einsum('xliORzda,LIORda->xlLiIz',bL11,tens11_d)
        return bL11

    def get_bL01(bI1,tens01_u,tens01_d,bR0):
        # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])
        bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        bL01 = np.einsum('xiIrRz,liorda->xlIoRzda',bL01,tens01_u)
        bL01 = np.einsum('xlIoRzda,LIORda->xlLoOz',bL01,tens01_d)
        return bL01

    def get_bL00(bL0,tens00_u,tens00_d,bI0):
        # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])
        bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        bL00 = np.einsum('xlLiIz,liorda->xLIorzda',bL00,tens00_u)
        bL00 = np.einsum('xLIorzda,LIORda->xoOrRz',bL00,tens00_d)
        return bL00

    def get_bL10_g(bL1,tens10_u,tens10_d,bO0):
        # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])
        bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        bL10 = np.einsum('wlLoOy,liorda->wLiOryda',bL10,tens10_u)
        bL10 = np.einsum('wLiOrydA,LIORda->wiIrRyaA',bL10,tens10_d)
        return bL10
 
    def get_bL11_g(bO1,tens11_u,tens11_d,bR1):
        # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])
        bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        bL11 = np.einsum('xoOrRz,liorda->xliORzda',bL11,tens11_u)
        bL11 = np.einsum('xliORzda,LIORdA->xlLiIzaA',bL11,tens11_d)
        return bL11

    def get_bL01_g(bI1,tens01_u,tens01_d,bR0):
        # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])
        bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        bL01 = np.einsum('xiIrRz,liorda->xlIoRzda',bL01,tens01_u)
        bL01 = np.einsum('xlIoRzda,LIORdA->xlLoOzaA',bL01,tens01_d)
        return bL01

    def get_bL00_g(bL0,tens00_u,tens00_d,bI0):
        # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])
        bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        bL00 = np.einsum('xlLiIz,liorda->xLIorzda',bL00,tens00_u)
        bL00 = np.einsum('xLIorzda,LIORdA->xoOrRzaA',bL00,tens00_d)
        return bL00


    ### order contractions assuming 'rol' mpo connectivity ###
    ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
    ###                               10 < 00 ~ 11 < 01 for 3-body operator
    if x_idx == (0,0):

        su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1,3])
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1,3])
        su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[])
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[])
        su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0,2])
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0,2])

        ## 00 corner 
        su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[])

        bL00 = get_bL00_g(bL[0],su00,sd00,bI[0])

        if iso_leg == 'r':  # update (0,0)-(0,1) bond
            bL10 = get_bL10(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])
            bL01 = get_bL01_g(bI[1],su01,sd01,bR[0])

            bLs = np.einsum('wiIrRx,xrRoOy->wiIoOy',bL10,bL11)
            bLs = np.einsum('wiIoOy,zlLoOyaA->wiIlLzaA',bLs,bL01)
            return np.einsum('xoOrRzaA,xoOrRzbB->ABab',bL00,bLs)

        elif iso_leg == 'o':  # update (0,0)-(1,0) bond
            bL10 = get_bL10_g(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])

            bLs = np.einsum('zlLoOy,xrRoOy->xlLrRz',bL01,bL11)
            bLs = np.einsum('wiIrRxaA,xlLrRz->wiIlLzaA',bL10,bLs)
            return np.einsum('xoOrRzaA,xoOrRzbB->ABab',bL00,bLs)


    elif x_idx == (0,1):

        su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[])
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[])
        su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[0,1])
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[0,1])
        su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2,3])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2,3])

        ## 01 corner
        su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[])
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[])

        bL01 = get_bL01_g(bI[1],su01,sd01,bR[0])

        if iso_leg == 'o':   # update (0,1)-(1,1) bond
            bL10 = get_bL10(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11_g(bO[1],su11,sd11,bR[1])
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])

            bLs = np.einsum('xoOrRw,xoOlLy->wrRlLy',bL00,bL10)
            bLs = np.einsum('wrRlLy,ylLiIzaA->wrRiIzaA',bLs ,bL11)
            return np.einsum('xlLoOzbB,xlLoOzaA->ABab',bLs,bL01)

        elif iso_leg == 'l':  # update (0,1)-(0,0) bond
            bL10 = get_bL10(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])
            bL00 = get_bL00_g(bL[0],su00,sd00,bI[0])

            bLs = np.einsum('xoOlLy,ylLiIz->xoOiIz',bL10,bL11)
            bLs = np.einsum('xoOrRwaA,xoOiIz->wrRiIzaA',bL00,bLs)
            return np.einsum('wlLoOzbB,wlLoOzaA->ABab',bLs,bL01)
        

    elif x_idx == (1,0):

        su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[])
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[])
        su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[0,1])
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[0,1])
        su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2,3])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2,3])

        ## 10 corner
        su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[])
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[])

        bL10 = get_bL10_g(bL[1],su10,sd10,bO[0])

        if iso_leg == 'r':     # update (1,0) - (1,1) bond
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])
            bL11 = get_bL11_g(bO[1],su11,sd11,bR[1])

            bLs = np.einsum('woOrRx,xrRiIy->woOiIy',bL00,bL01)
            bLs = np.einsum('woOiIy,zlLiIyaA->woOlLzaA',bLs, bL11)
            return np.einsum('xiIrRzaA,xiIrRzbB->ABab',bL10,bLs)

        elif iso_leg == 'i':   # update (1,0) - (0,0) bond 
            bL00 = get_bL00_g(bL[0],su00,sd00,bI[0])
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])
            bL11 = get_bL11(bO[1],su11,sd11,bR[1])

            bLs = np.einsum('xrRiIy,zlLiIy->xrRlLz',bL01,bL11)
            bLs = np.einsum('woOrRxaA,xrRlLz->woOlLzaA',bL00,bLs)
            return np.einsum('xiIrRzaA,xiIrRzbB->ABab',bL10,bLs)


    elif x_idx == (1,1):

        su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1,3])
        sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1,3])
        su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[])
        sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[])
        su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0,2])
        sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0,2])

        ## 11 corner
        su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[])
        sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[])

        bL11 = get_bL11_g(bO[1],su11,sd11,bR[1])

        if iso_leg == 'l':      # update (1,1) - (1,0)
            bL01 = get_bL01(bI[1],su01,sd01,bR[0])
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])
            bL10 = get_bL10_g(bL[1],su10,sd10,bO[0])

            bLs = np.einsum('xiIlLy,ylLoOz->xiIoOz',bL00,bL01)
            bLs = np.einsum('xiIrRwaA,xiIoOz->wrRoOzaA',bL10,bLs)

            return np.einsum('xlLiIzbB,xlLiIzaA->ABab',bLs,bL11)

        elif iso_leg == 'i':    # update (1,1) - (0,1)
            bL01 = get_bL01_g(bI[1],su01,sd01,bR[0])
            bL00 = get_bL00(bL[0],su00,sd00,bI[0])
            bL10 = get_bL10(bL[1],su10,sd10,bO[0])

            bLs = np.einsum('xiIrRw,xiIlLy->wrRlLy',bL10,bL00)
            bLs = np.einsum('wrRlLy,ylLoOzaA->wrRoOzaA',bLs, bL01)
            return np.einsum('xlLiIzbB,xlLiIzaA->ABab',bLs,bL11)

    else:  raise (IndexError)




def embed_sites_oog(sub_tldm_u, sub_tldm_d, env_list, decompose=False, idx_list=None, idx_conns=None,
                    decompose_g=False):
    ''' env d/dg <approx|exact> 
        exact ansatz has disentangler g
        decompose:  True--decomposed into 2 site unitaries (one per bond)
                    False--one giant unitary for all sites
                    (only affects 2x2 sites)
    '''

    L1,L2 = sub_tldm_u.shape
    bL, bI, bO, bR = env_list

    if (L1,L2) == (1,2):   # horizontal trotter step
    
        ## update site 1
        s1u = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[3])
        s1d = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[3])
        
        env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
        env_block1 = np.einsum('wlLiIy,liorda->wLIorday',env_block1,s1u)
        env_block1 = np.einsum('wLIorday,LIORdA->woOrRyaA',env_block1,s1d)
        env_block1 = np.einsum('woOrRyaA,yoOz->wrRzaA',env_block1,bO[0])

        ## site 2 boundary
        s2u = PEPX_GL.get_site(sub_tldm_u,(0,1))
        s2d = PEPX_GL.get_site(sub_tldm_d,(0,1))

        ## grad (missing bra)
        env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
        env_block2 = np.einsum('liorda,wiIrRy->wlIoRyda',s2u,env_block2)
        env_block2 = np.einsum('LIORdA,wlIoRyda->wlLoOyaA',s2d,env_block2)
        env_block2 = np.einsum('wlLoOyaA,zoOy->wlLzaA',env_block2,bO[1])

        return np.einsum('wlLzaA,wlLzbB->ABab',env_block1,env_block2)

    elif (L1,L2) == (2,1):   # vertical trotter step
    
        ## update site 1
        s1u = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2])
        s1d = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2])
        
        env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
        env_block1 = np.einsum('wlLiIy,liorda->wLIorday',env_block1,s1u)   # assume sum(shape(s1u)) < sum(sh(s2u))
        env_block1 = np.einsum('wLIorday,LIORdA->woOrRyaA',env_block1,s1d)
        env_block1 = np.einsum('woOrRyaA,yrRz->woOzaA',env_block1,bR[0])

        ## site 2 boundary
        s2u = PEPX_GL.get_site(sub_tldm_u,(1,0))
        s2d = PEPX_GL.get_site(sub_tldm_d,(1,0))
    
        env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
        env_block2 = np.einsum('wlLoOy,liorda->wLiOryda',env_block2,s2u)
        env_block2 = np.einsum('wLiOryda,LIORdA->wiIrRyaA',env_block2,s2d)
        env_block2 = np.einsum('wiIrRyaA,zrRy->wiIzaA',env_block2,bR[1])

        return np.einsum('wiIzaA,wiIzbB->ABab',env_block1,env_block2)   

    elif (L1,L2) == (2,2):  # LR/square trotter step

        def get_bL10_g(bL1,tens10_u,tens10_d,bO0):
            # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])
            bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
            bL10 = np.einsum('wlLoOy,liorda->wLiOryda',bL10,tens10_u)
            bL10 = np.einsum('wLiOrydA,LIORda->wiIrRyaA',bL10,tens10_d)
            return bL10
 
        def get_bL11_g(bO1,tens11_u,tens11_d,bR1):
            # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])
            bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
            bL11 = np.einsum('xoOrRz,liorda->xliORzda',bL11,tens11_u)
            bL11 = np.einsum('xliORzda,LIORdA->xlLiIzaA',bL11,tens11_d)
            return bL11

        def get_bL01_g(bI1,tens01_u,tens01_d,bR0):
            # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])
            bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
            bL01 = np.einsum('xiIrRz,liorda->xlIoRzda',bL01,tens01_u)
            bL01 = np.einsum('xlIoRzda,LIORdA->xlLoOzaA',bL01,tens01_d)
            return bL01

        def get_bL00_g(bL0,tens00_u,tens00_d,bI0):
            # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])
            bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
            bL00 = np.einsum('xlLiIz,liorda->xLIorzda',bL00,tens00_u)
            bL00 = np.einsum('xLIorzda,LIORdA->xoOrRzaA',bL00,tens00_d)
            return bL00


        ### order contractions assuming 'rol' mpo connectivity ###
        ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
        ###                               10 < 00 ~ 11 < 01 for 3-body operator
        if decompose_g:   # 4 2-site unitaries on (0,0)-roli bonds
            
            g_envs = []
            for ind in len(idx_list):
               x_idx = idx_list[ind]
               iso_leg = idx_conns[ind]
               g_envs.append( embed_sites_oog_2x2_bond(sub_tldm_u, sub_tldm_d, env_list, x_idx, iso_leg) )

            return g_envs

        else:

            su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1,3])
            sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1,3])
            su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[])
            sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[])
            su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0,2])
            sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0,2])
            su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[])
            sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[])

            bL00 = get_bL00_g(bL[0],su00,sd00,bI[0])
            bL10 = get_bL10_g(bL[1],su10,sd10,bO[0])
            bL11 = get_bL11_g(bO[1],su11,sd11,bR[1])
            bL01 = get_bL01_g(bI[1],su01,sd01,bR[0])

            bLs = np.einsum('xiIlLwaA,xiIrRydD->wlLrRyaAdD',bL00,bL10)
            bLs = np.einsum('wlLrRyaAdD,yrRoOzcC->wlLoOzaAcCdD',bLs,bL11)
            bLs = np.einsum('wlLoOzaAcCdD,wlLoOzbB->ABCDabcd',bLs,bL01)

            return bLs

    else:
        raise (NotImplementedError)


def embed_sites_xog(sub_tldm_u,sub_tldm_d,envs,x_idx,g_opt,qr_reduce=False,iso_leg=None,qu_idx=None,qd_idx=None,
                    decompose_g=False,idx_list=None,idx_conns=None):
    # assume sum(shape(bra)) < sum(shape(ket))
    # g_opt: site1 a(bra),a(ket), site2.. 

    L1,L2 = sub_tldm_u.shape
    bL, bI, bO, bR = envs

    return_all = False 
    if qr_reduce:
        tens_u_idx = PEPX_GL.get_site(sub_tldm_u,x_idx)
        tens_d_idx = PEPX_GL.get_site(sub_tldm_d,x_idx)
        
        if qu_idx is None or qd_idx is None:
            qu_idx, ru_idx, axT_inv = PEPX.QR_factor(tens_u_idx,iso_leg,d_end=True)
            qd_idx, rd_idx, axT_inv = PEPX.QR_factor(tens_d_idx,iso_leg,d_end=True)
            # print 'red env axT', axT_inv
            return_all = True


    if (L1,L2) == (1,2):   # horizontal trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0])
            s2d = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0])
    
            env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
            env_block2 = np.einsum('wiIrRy,liorda->wlIoRday',env_block2,s2u)
            env_block2 = np.einsum('wlIoRday,LIORdA->wlLoOyaA',env_block2,s2d)
            env_block2 = np.einsum('wlLoOybB,ABab->wlLoOyaA',env_block2,g_opt) 
            env_block2 = np.einsum('wlLoOyaA,zoOy->wlLzaA',env_block2,bO[1])
    
            ## site 1 boundary
            s1d = PEPX_GL.get_site(sub_tldm_d,(0,0))

            ## grad (missing bra)
            if qr_reduce:
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lioq->wLIoqy',env_block1,qu_idx)
                env_block1 = np.einsum('wLIoqy,LIORdA->woORqdAy',env_block1,s1d)
                env_block1 = np.einsum('woORqdAy,yoOz->wRqdAz',env_block1,bO[0])
                env_out = np.einsum('wRqdAz,wrRzaA->qrda',env_block1,env_block2)
            else:
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,LIORdA->wliORdAy',env_block1,s1d)
                env_block1 = np.einsum('wliORdAy,yoOz->wlioRdAz',env_block1,bO[0])
                env_out = np.einsum('wlioRdAz,wrRzaA->liorda',env_block1,env_block2)
 

        ##### possibly a bug somewhere here? #######
        elif x_idx == (0,1):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[3])
            s1d = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[3])
        
            env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liorda->wLIorday',env_block1,s1u)
            env_block1 = np.einsum('wLIorday,LIORdA->woOrRyaA',env_block1,s1d)
            env_block1 = np.einsum('woOrRyaA,ABab->woOrRybB',env_block1,g_opt)
            env_block1 = np.einsum('woOrRybB,yoOz->wrRzbB',env_block1,bO[0])

            ## site 2 boundary
            s2d = PEPX_GL.get_site(sub_tldm_d,(0,1))

            ## grad (missing bra)
            if qr_reduce:
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('iorq,wiIrRy->wqIoRy',qu_idx,env_block2)
                env_block2 = np.einsum('LIORdA,wqIoRy->wqLoOydA',s2d,env_block2)
                env_block2 = np.einsum('wqLoOydA,zoOy->wqLdAz',env_block2,bO[1])
                env_out = np.einsum('wlLzaA,wqLdAz->qlda',env_block1,env_block2)
            else:
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('LIORdA,wiIrRy->wLiOrydA',s2d,env_block2)
                env_block2 = np.einsum('wLiOrydA,zoOy->wLiorzdA',env_block2,bO[1])
                env_out = np.einsum('wlLzaA,wLiorzdA->liorda',env_block1,env_block2)

        else:  raise (IndexError)
        

    elif (L1,L2) == (2,1):   # vertical trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1])
            s2d = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1])
    
            env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
            env_block2 = np.einsum('wlLoOy,liorda->wLiOrday',env_block2,s2u)
            env_block2 = np.einsum('wLiOrday,LIORdA->wiIrRyaA',env_block2,s2d)
            env_block2 = np.einsum('wiIrRybB,ABab->wiIrRyaA',env_block2,g_opt)
            env_block2 = np.einsum('wiIrRyaA,zrRy->wiIzaA',env_block2,bR[1])

            ## site 1 boundary
            s1d = PEPX_GL.get_site(sub_tldm_d,(0,0))
    
            if qr_reduce:
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lirq->wLIqry',env_block1,qu_idx)
                env_block1 = np.einsum('wLIqry,LIORdA->wqOrRdAy',env_block1,s1d)
                env_block1 = np.einsum('wqOrRdAy,yrRz->wqOdAz',env_block1,bR[0])
                env_out = np.einsum('wqOdAz,woOzaA->qoda',env_block1,env_block2)
            else:
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,LIORdA->wliORdAy',env_block1,s1d)
                env_block1 = np.einsum('wliORdAy,yrRz->wliOrdAz',env_block1,bR[0])
                env_out = np.einsum('wliOrdAz,woOzaA->liorda',env_block1,env_block2)
    
        elif x_idx == (1,0):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2])
            s1d = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2])
        
            env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liorda->wLIorday',env_block1,s1u)   # assume sum(shape(s1u)) < sum(sh(s2u))
            env_block1 = np.einsum('wLIorday,LIORdA->woOrRyaA',env_block1,s1d)
            env_block1 = np.einsum('woOrRyaA,ABab->woOrRybB',env_block1,g_opt)
            env_block1 = np.einsum('woOrRybB,yrRz->woOzbB',env_block1,bR[0])

            ## site 2 boundary
            s2d = PEPX_GL.get_site(sub_tldm_d,(1,0))
    
            if qr_reduce:
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,lorq->wLqOry',env_block2,qu_idx)
                env_block2 = np.einsum('wLqOry,LIORdA->wqIrRdAy',env_block2,s2d)
                env_block2 = np.einsum('wqIrRdAy,zrRy->wqIdAz',env_block2,bR[1])
                env_out = np.einsum('wiIzaA,wqIdAz->qida',env_block1,env_block2)
            else:
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,LIORdA->wlIoRdAy',env_block2,s2d)
                env_block2 = np.einsum('wlIoRdAy,zrRy->wlIordAz',env_block2,bR[1])
                env_out = np.einsum('wiIzaA,wlIordAz->liorda',env_block1,env_block2)

        else:  raise (IndexError)
     
   
    elif (L1,L2) == (2,2):  # LR/square trotter step

        if decompose_g:   # 4 2-site unitaries on (0,0)-roli bonds
            
            return embed_sites_xog_2x2_bond(sub_tldm_u, sub_tldm_d, env_list, g_opt, idx_list, idx_conns)

        else:

            def get_bL10_g(bL1,tens10_u,tens10_d,bO0):
                # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])
                bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                bL10 = np.einsum('wlLoOy,liorda->wLiOryda',bL10,tens10_u)
                bL10 = np.einsum('wLiOrydA,LIORda->wiIrRyaA',bL10,tens10_d)
                return bL10
 
            def get_bL11_g(bO1,tens11_u,tens11_d,bR1):
                # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                bL11 = np.einsum('xoOrRz,liorda->xliORzda',bL11,tens11_u)
                bL11 = np.einsum('xliORzda,LIORdA->xlLiIzaA',bL11,tens11_d)
                return bL11

            def get_bL01_g(bI1,tens01_u,tens01_d,bR0):
                # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                bL01 = np.einsum('xiIrRz,liorda->xlIoRzda',bL01,tens01_u)
                bL01 = np.einsum('xlIoRzda,LIORdA->xlLoOzaA',bL01,tens01_d)
                return bL01

            def get_bL00_g(bL0,tens00_u,tens00_d,bI0):
                # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                bL00 = np.einsum('xlLiIz,liorda->xLIorzda',bL00,tens00_u)
                bL00 = np.einsum('xLIorzda,LIORdA->xoOrRzaA',bL00,tens00_d)
                return bL00


            ## g_opt (not decomposed):  ABCDabcd (where ABCD = (0,0)-(0,1)-(1,1)-(1,0))

            ### order contractions assuming 'rol' mpo connectivity ###
            ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
            ###                               10 < 00 ~ 11 < 01 for 3-body operator
            if x_idx == (0,0):

                su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1,3])
                sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1,3])
                su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[])
                sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[])
                su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0,2])
                sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0,2])

                ## 00 corner 
                sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[])

                bL10 = get_bL10_g(bL[1],su10,sd10,bO[0])
                bL11 = get_bL11_g(bO[1],su11,sd11,bR[1])
                bL01 = get_bL01_g(bI[1],su01,sd01,bR[0])

                bLs = np.einsum('wiIrRxdD,xrRoOycC->wiIoOycCdD',bL10,bL11)
                bLs = np.einsum('wiIoOycCdD,ABCDabcd->wiIoOyaAbB',bLs,g_opt)
                bLs = np.einsum('wiIoOyaAbB,zlLoOybB->wiIlLzaA',bLs,bL01)

                if qr_reduce:
                    bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                    bL00 = np.einsum('xlLiIz,lioq->xLIoqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIoqz,LIORdA->xoOqRzdA',bL00,sd00)
                    env_out = np.einsum('xoOqRzdA,xoOrRzaA->qrda',bL00,bLs)
                else:
                    bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                    bL00 = np.einsum('xlLiIz,LIORdA->xliORdzA',bL00,sd00)
                    env_out = np.einsum('xliORdzA,xoOrRzaA->liorda',bL00,bLs)
             
            elif x_idx == (0,1):

                su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[])
                sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[])
                su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[0,1])
                sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[0,1])
                su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2,3])
                sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2,3])

                ## 01 corner
                sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[])

                bL10 = get_bL10_g(bL[1],su10,sd10,bO[0])
                bL11 = get_bL11_g(bO[1],su11,sd11,bR[1])
                bL00 = get_bL00_g(bL[0],su00,sd00,bI[0])

                bLs = np.einsum('xoOrRwaA,xoOlLydD->wrRlLyaAdD',bL00,bL10)
                bLs = np.einsum('wrRlLyaAdD,ABCDabcd->wrRlLybBcC',bLs,g_opt)
                bLs = np.einsum('wrRlLybBcC,ylLiIzcC->wrRiIzbB',bLs ,bL11)

                if qr_reduce:
                    bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                    bL01 = np.einsum('xiIrRz,lirq->xlIqRz',bL01,qu_idx)
                    bL01 = np.einsum('xlIqRz,LIORdA->xlLqOzdA',bL01,sd01)
                    env_out = np.einsum('xlLqOzdA,xlLoOzaA->qoda',bL01,bLs)
                else:
                    bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                    bL01 = np.einsum('xiIrRz,LIORdA->xLiOrdAz',bL01,sd01)
                    env_out = np.einsum('xlLoOzaA,xLiOrdAz->liorda',bLs,bL01)

            elif x_idx == (1,0):

                su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[])
                sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[])
                su11 = PEPX_GL.get_site(sub_tldm_u,(1,1),no_lam=[0,1])
                sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[0,1])
                su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[2,3])
                sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[2,3])

                ## 10 corner
                sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[])

                bL00 = get_bL00_g(bL[0],su00,sd00,bI[0])
                bL01 = get_bL01_g(bI[1],su01,sd01,bR[0])
                bL11 = get_bL11_g(bO[1],su11,sd11,bR[1])

                bLs = np.einsum('woOrRxaA,xrRiIybB->woOiIyaAbB',bL00,bL01)
                bLs = np.einsum('woOiIyaAbB,ABCDabcd->woOiIycCdD',bLs,g_opt)
                bLs = np.einsum('woOiIycCdD,zlLiIycC->woOlLzdD',bLs, bL11)

                if qr_reduce:
                    bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                    bL10 = np.einsum('xlLoOz,lioq->xLiOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLiOqz,LIORdA->xiIqRzdA',bL10,sd10)
                    env_out = np.einsum('xiIqRzdA,xiIrRzaA->qrda',bL10,bLs)
                else:
                    bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                    bL10 = np.einsum('xlLoOz,LIORdA->xlIoRdAz',bL10,sd10)
                    env_out = np.einsum('xlIoRdAz,xiIrRzaA->liorda',bL10,bLs)

            elif x_idx == (1,1):

                su10 = PEPX_GL.get_site(sub_tldm_u,(1,0),no_lam=[1,3])
                sd10 = PEPX_GL.get_site(sub_tldm_d,(1,0),no_lam=[1,3])
                su00 = PEPX_GL.get_site(sub_tldm_u,(0,0),no_lam=[])
                sd00 = PEPX_GL.get_site(sub_tldm_d,(0,0),no_lam=[])
                su01 = PEPX_GL.get_site(sub_tldm_u,(0,1),no_lam=[0,2])
                sd01 = PEPX_GL.get_site(sub_tldm_d,(0,1),no_lam=[0,2])

                ## 11 corner
                sd11 = PEPX_GL.get_site(sub_tldm_d,(1,1),no_lam=[])

                bL01 = get_bL01_g(bI[1],su01,sd01,bR[0])
                bL00 = get_bL00_g(bL[0],su00,sd00,bI[0])
                bL10 = get_bL10_g(bL[1],su10,sd10,bO[0])

                bLs = np.einsum('xiIlLyaA,ylLoOzbB->xiIoOzaAbB',bL00,bL01)
                bLs = np.einsum('xiIoOzaAbB,ABCDabcd->xiIoOzcCdD',bLs,g_opt)
                bLs = np.einsum('xiIrRwdD,xiIoOzcCdD->wrRoOzcC',bL10,bLs)

                if qr_reduce:
                    bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                    bL11 = np.einsum('xoOrRz,iorq->xqiORz',bL11,qu_idx)
                    bL11 = np.einsum('xqiORz,LIORdA->xqLiIzdA',bL11,sd11)
                    env_out = np.einsum('xqLiIzdA,xlLiIzaA->qlda',bL11,bLs)
                else:
                    bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                    bL11 = np.einsum('xoOrRz,LIORdA->xLIordAz',bL11,sd11)
                    env_out = np.einsum('xlLiIzaA,xLIordAz->liorda',bLs,bL11)


            else:  raise (IndexError)

    else:
        raise (NotImplementedError)

    if return_all:      
        return env_out, qu_idx, qd_idx, axT_inv
    else:
        return env_out
           

# @profile
def embed_sites_xx(sub_tldm_u, sub_tldm_d, envs, x_idx):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''

    return ENV_GL.embed_sites(PEPX_GL.flatten(sub_tldm_u), PEPX_GL.flatten(sub_tldm_d), envs, x_idx, 'xx')


def embed_sites_xo(sub_tldm_u, sub_tldm_d, envs, x_idx):

    dbs = sub_tldm_u[0,0].shape[4:]
    site_xo = ENV_GL.embed_sites(PEPX_GL.flatten(sub_tldm_u), PEPX_GL.flatten(sub_tldm_d), envs, x_idx, 'xo')
    return site_xo.reshape(site_xo.shape[:4]+dbs)


def embed_sites_norm(sub_pepx,envs_list):
    ''' get norm of sub_pepx embedded in env '''

    return ENV_GL.embed_sites_norm(PEPX_GL.flatten(sub_pepx),envs_list)


def embed_sites_ovlp(sub_tldm_u, sub_tldm_d, envs_list):

    # print [m.shape for idx,m in np.ndenumerate(sub_tldm_u)]
    # print sub_tldm_u.phys_bonds
    # print [m.shape for idx,m in np.ndenumerate(sub_tldm_d)]
    # print sub_tldm_d.phys_bonds

    # temp1 = PEPX_GL.flatten(sub_tldm_u)
    # temp2 = PEPX_GL.flatten(sub_tldm_d)
    # print [m.shape for idx,m in np.ndenumerate(temp1)]
    # print temp1.phys_bonds
    # print [m.shape for idx,m in np.ndenumerate(temp2)]
    # print temp2.phys_bonds

    # exit()

    return ENV_GL.embed_sites_ovlp(PEPX_GL.flatten(sub_tldm_u), PEPX_GL.flatten(sub_tldm_d), envs_list)


def embed_sites_ovlp_g(sub_tldm_u, sub_tldm_d, envs_list, g_opt):
    ''' calc ovlp but where ket tldm has disentangler on ancilla '''

    g_env = embed_sites_oog(sub_tldm_u, sub_tldm_d, envs_list)
    return np.tensordot(g_env,g_opt,axes=(range(g_env.ndim),range(g_opt.ndim)))


#############################################################
####     reduced embedded methods (on the fly env)       ####
#############################################################


# @profile
def red_embed_sites_xx(sub_tldm_u, sub_tldm_d, envs, x_idx, iso_leg, qu_idx=None, qd_idx=None):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''
    return ENV_GL.red_embed_sites(PEPX_GL.flatten(sub_tldm_u), PEPX_GL.flatten(sub_tldm_d), 
                                  envs, x_idx, iso_leg, 'xx', qu_idx, qd_idx)


def red_embed_sites_xo(sub_tldm_u, sub_tldm_d, envs, x_idx, iso_leg, qu_idx=None, qd_idx=None):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''
    dbs = sub_tldm_u[0,0].shape[4:]
    site_xo = ENV_GL.red_embed_sites(PEPX_GL.flatten(sub_tldm_u), PEPX_GL.flatten(sub_tldm_d), 
                                  envs, x_idx, iso_leg, 'xo', qu_idx, qd_idx)
    return site_xo.reshape(site_xo.shape[:4]+dbs)


def red_embed_sites_xog(sub_tldm_u, sub_tldm_d, envs, x_idx, g_opt, iso_leg, qu_idx=None, qd_idx=None):

    return embed_sites_xog(sub_tldm_u, sub_tldm_d, envs, x_idx, g_opt, True, iso_leg, qu_idx, qd_idx)
