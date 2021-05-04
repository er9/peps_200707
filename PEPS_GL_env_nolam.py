# import warnings
# warnings.filterwarnings("error")

import numpy as np
import time
import tens_fcts as tf

import MPX
import PEPX
import PEPX_GL
import PEPS_env as ENV


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

def get_next_boundary_O(peps_u_row,peps_d_row,boundary,XMAX=100):
    '''  get outside boundary (facing in)
         note:  peps_x_row is 2-D, but a 1 x L2 pepx_GL
    '''
    
    L2 = len(peps_u_row)
    boundary_mpo = boundary.copy()

    for j in range(L2):
        x_lam = [0,1]
        ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # tens2 = np.einsum('liord,LIORd,xoOy->xlLiIyrR',ptens_u,ptens_d,boundary_mpo[j])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens2 = np.einsum('LIORd,xoOy->xLIoyRd',ptens_d,boundary_mpo[j])
            tens2 = np.einsum('liord,xLIoyRd->xlLiIyrR',ptens_u,tens2)
        else:
            tens2 = np.einsum('liord,xoOy->xliOyrd',ptens_u,boundary_mpo[j])
            tens2 = np.einsum('LIORd,xliOyrd->xlLiIyrR',ptens_d,tens2)
        boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

    # temp = boundary.copy()
    # for j in range(L2):

    #     x_lam = [0,2]
    #     ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

    #     temp[j] = np.einsum('ludr,Uu,Dd->lUDr',temp[j],np.diag(np.sqrt(peps_u_row.lambdas[j,2])),
    #                                                    np.diag(np.sqrt(peps_d_row.lambdas[j,2])) )
    #     
    #     # tens2 = np.einsum('liord,LIORd,xoOy->xlLiIyrR',ptens_u,ptens_d,boundary_mpo[j])
    #     tens2 = np.einsum('liord,xoOy->xliOyrd',ptens_u,temp[j])
    #     tens2 = np.einsum('LIORd,xliOyrd->xlLiIyrR',ptens_d,tens2)
    #     temp[j] = tf.reshape(tens2,'iii,i,i,iii')

    #     print 'bonds', peps_u_row.lambdas[j]
    #     temp[j] = np.einsum('ludr,Uu,Dd->lUDr',temp[j],np.diag(1./np.sqrt(peps_u_row.lambdas[j,1])),
    #                                                    np.diag(1./np.sqrt(peps_d_row.lambdas[j,1])) )

    # 
    # diff_boundary = temp-boundary_mpo
    # print 'diff', np.linalg.norm( diff_boundary.getSites() )
 
    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    if np.max(err) > 1.0e-1:
        print [np.linalg.norm(m) for m in boundary_mpo]
        print err
        raise RuntimeWarning('bound O large compression error, XMAX'+str(XMAX))

    return boundary_mpo, err


def get_next_boundary_I(peps_u_row,peps_d_row,boundary,XMAX=100):
    ''' get inside boundary (facing out)
        note:  peps_x_row is 1-D
    '''

    L2 = len(peps_u_row)
    boundary_mpo = boundary.copy()

    for j in range(L2):

        x_lam = [2,3]
        ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # print 'bI', np.linalg.norm(ptens_u), np.linalg.norm( PEPX_GL.get_site(peps_u_row,j,no_lam=[1]))

        # tens2 = np.einsum('xiIy,liord,LIORd->xlLoOyrR',boundary_mpo[j],peps_u_row[j],peps_d_row[j])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens2 = np.einsum('xiIy,LIORd->xLiOyRd',boundary_mpo[j],ptens_d)
            tens2 = np.einsum('xLiOyRd,liord->xlLoOyrR',tens2,ptens_u)
        else:
            tens2 = np.einsum('xiIy,liord->xlIoyrd',boundary_mpo[j],ptens_u)
            tens2 = np.einsum('xlIoyrd,LIORd->xlLoOyrR',tens2,ptens_d)
        boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

    # temp = boundary.copy()
    # for j in range(L2):

    #     x_lam = [1,3]
    #     ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

    #     temp[j] = np.einsum('ludr,Uu,Dd->lUDr',temp[j],np.diag(np.sqrt(peps_u_row.lambdas[j,1])),
    #                                                    np.diag(np.sqrt(peps_d_row.lambdas[j,1])) )
    #     
    #     # tens2 = np.einsum('liord,LIORd,xoOy->xlLiIyrR',ptens_u,ptens_d,boundary_mpo[j])
    #     tens2 = np.einsum('xiIy,liord->xlIoyrd',temp[j],ptens_u)
    #     tens2 = np.einsum('xlIoyrd,LIORd->xlLoOyrR',tens2,ptens_d)
    #     temp[j] = tf.reshape(tens2,'iii,i,i,iii')

    #     print 'bonds', peps_u_row.lambdas[j]
    #     temp[j] = np.einsum('ludr,Uu,Dd->lUDr',temp[j],np.diag(1./np.sqrt(peps_u_row.lambdas[j,2])),
    #                                                    np.diag(1./np.sqrt(peps_d_row.lambdas[j,2])) )

    # 
    # diff_boundary = temp-boundary_mpo
    # print 'diff', np.linalg.norm( diff_boundary.getSites() )
    
    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,0)

    # print 'bi', [np.linalg.norm(m) for m in boundary_mpo]

    if np.max(err) > 1.0e-1:
        print [np.linalg.norm(m) for m in boundary_mpo]
        print err
        raise RuntimeWarning('bound I large compression error, XMAX'+str(XMAX))

    return boundary_mpo, err


def get_next_boundary_L(peps_u_col,peps_d_col,boundary,XMAX=100):
    ''' get left boundary (facing right)
        note:  peps_x_col is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):

        x_lam = [1,3]
        ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')

        # tens2 = np.einsum('xlLy,liord,LIORd->xiIrRyoO',boundary_mpo[i],peps_u_col[i],peps_d_col[i])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens2 = np.einsum('xlLy,LIORd->xlIORyd',boundary_mpo[i],ptens_d)
            tens2 = np.einsum('xlIORyd,liord->xiIrRyoO',tens2,ptens_u)
        else:
            tens2 = np.einsum('xlLy,liord->xLioryd',boundary_mpo[i],ptens_u)
            tens2 = np.einsum('xLioryd,LIORd->xiIrRyoO',tens2,ptens_d)
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')
    
    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err  = MPX.compress_reg(boundary_mpo,XMAX,0)

    if np.max(err) > 1.0e-1:
        print [np.linalg.norm(m) for m in boundary_mpo]
        print err
        raise RuntimeWarning('bound L large compression error, XMAX'+str(XMAX))

    return boundary_mpo, err


def get_next_boundary_R(peps_u_col,peps_d_col,boundary,XMAX=100):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):

        x_lam = [0,2]
        ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')


        # tens2 = np.einsum('liord,LIORd,xrRy->xiIlLyoO',peps_u_col[i],peps_d_col[i],boundary_mpo[i])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens2 = np.einsum('LIORd,xrRy->xLIOryd',ptens_d,boundary_mpo[i])
            tens2 = np.einsum('liord,xLIOryd->xiIlLyoO',ptens_u,tens2)
        else:
            tens2 = np.einsum('liord,xrRy->xlioRyd',ptens_u,boundary_mpo[i])
            tens2 = np.einsum('LIORd,xlioRyd->xiIlLyoO',ptens_d,tens2)
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')

    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

    if np.max(err) > 1.0e-1:
        print [np.linalg.norm(m) for m in boundary_mpo]
        print err
        raise RuntimeWarning('bound R large compression error, XMAX'+str(XMAX))

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

def get_next_subboundary_O(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100,c_tol=1.0e-8):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()

    for j in range(L2):

        # if j == L2-1:     x_lam = [1] 
        # else:             x_lam = [1,3]

        if j == 0:     x_lam = [1] 
        else:          x_lam = [0,1]

        ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # tens = np.einsum('liord,LIORd,xoOy->lLxiIrRy', peps_u_row[j],peps_d_row[j],e_mpo[j])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens = np.einsum('LIORd,xoOy->LxIoRyd',ptens_d,e_mpo[j])
            tens = np.einsum('liord,LxIoRyd->lLxiIrRy',ptens_u,tens)
        else:
            tens = np.einsum('liord,xoOy->lxiOryd',ptens_u,e_mpo[j])
            tens = np.einsum('LIORd,lxiOryd->lLxiIrRy',ptens_d,tens)
        e_mpo[j] = tens  # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wlLx,lLxiIrRy->wiIrRy',boundL_tens,e_mpo[0])   # i(rRo) -- (lLx)oO(rRy)
    e_mpo[-1] = np.einsum('...rRy,zrRy->...z',e_mpo[-1],boundR_tens)      # (lLx)oO(rRy) -- i(lLo)

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')
    
    # test1 = MPX.canonicalize(e_mpo,1)
    # test2 = MPX.canonicalize(e_mpo,0)

    # print 'subbound O'
    # print MPX.norm( e_mpo - test1 )
    # print MPX.norm( e_mpo - test2 )
    # print MPX.norm( test2 - test1 )
    # print [np.linalg.norm(m) for m in e_mpo]
    # print [np.linalg.norm(m) for m in test1]
    # print [np.linalg.norm(m) for m in test2]
    # print '---'

    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,1) #,use_qr=True)   ### something wonky with this??  (svd version)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0,tol=c_tol)
    # err = [np.nan]*(L2-1)

    # print 'subbound O', err
    # test1, err = MPX.compress_reg(test1,XMAX,0)
    # test2, err = MPX.compress_reg(test2,XMAX,1)
    # print [np.linalg.norm(m) for m in test1]
    # print [np.linalg.norm(m) for m in test2]
    # print [np.linalg.norm(test1[i] - test2[i]) for i in range(len(test1))]
    # print '---'
    

    if len(err) > 0:
        if np.max(err) > 1.0e-1:
            print [np.linalg.norm(m) for m in e_mpo]
            print err
            raise RuntimeWarning('subbound O large compression error'+str(XMAX))

    return e_mpo, err


def get_next_subboundary_I(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100,c_tol=1.0e-8):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()
    
    for j in range(L2):

        # if j == 0:     x_lam = [2]
        # else:          x_lam = [0,2]

        if j == L2-1:     x_lam = [2]
        else:             x_lam = [2,3]

        ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # tens = np.einsum('xiIy,liord,LIORd->xlLoOyrR', e_mpo[j],peps_u_row[j],peps_d_row[j])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens = np.einsum('xiIy,LIORd->xLiOyRd',e_mpo[j],ptens_d)
            tens = np.einsum('xLiOyRd,liord->xlLoOyrR',tens,ptens_u)
        else:
            tens = np.einsum('xiIy,liord->xlIoyrd',e_mpo[j],ptens_u)
            tens = np.einsum('xlIoyrd,LIORd->xlLoOyrR',tens,ptens_d)
        e_mpo[j] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('xlLw,xlLoOyrR->woOyrR',boundL_tens,e_mpo[0])   # (irR)o -- (xlL)iI(yrR)
    e_mpo[-1] = np.einsum('...yrR,yrRz->...z',e_mpo[-1],boundR_tens)      # (xlL)iI(yrR) -- (ilL)o

    if L2 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for j in range(1,L2-1):   e_mpo[j] = tf.reshape(e_mpo[j],'iii,i,i,iii')

    # test1 = MPX.canonicalize(e_mpo,1)
    # test2 = MPX.canonicalize(e_mpo,0)

    # print 'subbound I'
    # print 't1', MPX.norm( e_mpo - test1 )   # nonnegligible neg number
    # print 't2', MPX.norm( e_mpo - test2 )
    # print 't1-t2', MPX.norm( test2 - test1 )
    # print [np.linalg.norm(m) for m in e_mpo]
    # print [np.linalg.norm(m) for m in test1]
    # print [np.linalg.norm(m) for m in test2]
    # print '---'
    
    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1,tol=c_tol)

    # print 'subbound I', err
    # test1, err = MPX.compress_reg(test1,XMAX,0)
    # test2, err = MPX.compress_reg(test2,XMAX,1)
    # print [np.linalg.norm(m) for m in test1]
    # print [np.linalg.norm(m) for m in test2]
    # print [np.linalg.norm(test1[i] - test2[i]) for i in range(len(test1))]
    # print '---'

    if len(err) > 0:
        if np.max(err) > 1.0e-1:
            print [np.linalg.norm(m) for m in e_mpo]
            print err
            raise RuntimeWarning('subbound I large compression error'+str(XMAX))

    return e_mpo, err


def get_next_subboundary_L(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100,c_tol=1.0e-8):

    L1 = len(peps_u_col)

    e_mpo = env_mpo.copy()
    
    for i in range(L1):

        # if i == L1-1:   x_lam = [3]
        # else:           x_lam = [2,3]   # looks right canonical

        if i == 0:   x_lam = [3]
        else:        x_lam = [1,3]   # looks right canonical

        ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')

        # if i == 0:
        #     ptens_u = PEPX_GL.apply_bond(ptens_u,1./np.sqrt(peps_u_col.lambdas[i,1]),1)
        #     ptens_d = PEPX_GL.apply_bond(ptens_d,1./np.sqrt(peps_d_col.lambdas[i,1]),1)
        # if i == L1-1:
        #     ptens_u = PEPX_GL.apply_bond(ptens_u,1./np.sqrt(peps_u_col.lambdas[i,2]),2)
        #     ptens_d = PEPX_GL.apply_bond(ptens_d,1./np.sqrt(peps_d_col.lambdas[i,2]),2)
       
        # # out bond
        # ptens_u = PEPX_GL.apply_bond(ptens_u,np.sqrt(peps_u_col.lambdas[i,3]),3)
        # ptens_d = PEPX_GL.apply_bond(ptens_d,np.sqrt(peps_d_col.lambdas[i,3]),3)
        # # in bond
        # ptens_u = PEPX_GL.apply_bond(ptens_u,1./np.sqrt(peps_u_col.lambdas[i,0]),0)
        # ptens_d = PEPX_GL.apply_bond(ptens_d,1./np.sqrt(peps_d_col.lambdas[i,0]),0)

        # print 'bL', np.linalg.norm(ptens_u)
        # print 'bL', np.linalg.norm(PEPX_GL.get_site(peps_u_col,i,no_lam=[]))

        # print 'sub L', e_mpo[i].shape, peps_u_col[i].shape, peps_d_col[i].shape
        # tens = np.einsum('xlLy,liord,LIORd->xiIrRyoO', e_mpo[i],peps_u_col[i],peps_d_col[i])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens = np.einsum('xlLy,LIORd->xIlRyOd',e_mpo[i],ptens_d)
            tens = np.einsum('xIlRyOd,liord->xiIrRyoO',tens,ptens_u)
        else:
            tens = np.einsum('xlLy,liord->xiLryod',e_mpo[i],ptens_u)
            tens = np.einsum('xiLryod,LIORd->xiIrRyoO',tens,ptens_d)
        e_mpo[i] = tens   # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    # print 'subL', boundI_tens.shape, e_mpo[0].shape
    e_mpo[0]  = np.einsum('xiIw,xiIrRyoO->wrRyoO',boundI_tens,e_mpo[0])    # l(oOr) -- (xiI)rR(yoO)
    e_mpo[-1] = np.einsum('...yoO,yoOz->...z',e_mpo[-1],boundO_tens)       # (xiI)rR(yoO) -- (liI)r


    if L1 > 1:
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')

    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1,tol=c_tol)

    if len(err) > 0:
        if np.max(err) > 1.0e-1:
            print [np.linalg.norm(m) for m in e_mpo]
            print [np.linalg.norm(m) for m in env_mpo], np.linalg.norm(boundI_tens), np.linalg.norm(boundO_tens)
            print [np.linalg.norm(m) for m in peps_u_col]
            print [np.linalg.norm(m) for m in peps_d_col]
            print err
            raise RuntimeWarning('subL large compression error, XMAX '+str(XMAX))

    return e_mpo, err


def get_next_subboundary_R(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100,c_tol=1.0e-8):

    L1 = len(peps_u_col)
    e_mpo = env_mpo.copy()



    for i in range(L1):

        # if i == 0:      x_lam = [0]
        # else:           x_lam = [0,1]           # looks left canonical

        if i == L1-1:      x_lam = [0]
        else:              x_lam = [0,2]

        ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')

        # if i == 0:
        #     ptens_u = PEPX_GL.apply_bond(ptens_u,1./np.sqrt(peps_u_col.lambdas[i,1]),1)
        #     ptens_d = PEPX_GL.apply_bond(ptens_d,1./np.sqrt(peps_d_col.lambdas[i,1]),1)
        # if i == L1-1:
        #     ptens_u = PEPX_GL.apply_bond(ptens_u,1./np.sqrt(peps_u_col.lambdas[i,2]),2)
        #     ptens_d = PEPX_GL.apply_bond(ptens_d,1./np.sqrt(peps_d_col.lambdas[i,2]),2)
       
        # # out bond
        # ptens_u = PEPX_GL.apply_bond(ptens_u,np.sqrt(peps_u_col.lambdas[i,0]),0)
        # ptens_d = PEPX_GL.apply_bond(ptens_d,np.sqrt(peps_d_col.lambdas[i,0]),0)
        # # in bond
        # ptens_u = PEPX_GL.apply_bond(ptens_u,1./np.sqrt(peps_u_col.lambdas[i,3]),3)
        # ptens_d = PEPX_GL.apply_bond(ptens_d,1./np.sqrt(peps_d_col.lambdas[i,3]),3)

        # print 'br',i,L1,e_mpo[i].shape, peps_u_col[i].shape, peps_d_col[i].shape
        # tens = np.einsum('liord,LIORd,xrRy->iIxlLoOy', peps_u_col[i],peps_d_col[i],e_mpo[i])
        if np.sum(ptens_u.shape) > np.sum(ptens_d.shape):
            tens = np.einsum('LIORd,xrRy->IxLrOyd',ptens_d,e_mpo[i])
            tens = np.einsum('liord,IxLrOyd->iIxlLoOy',ptens_u,tens)
        else:
            tens = np.einsum('liord,xrRy->ixlRoyd',ptens_u,e_mpo[i])
            tens = np.einsum('LIORd,ixlRoyd->iIxlLoOy',ptens_d,tens)
        e_mpo[i] = tens    # tf.reshape(tens,'iii,i,i,iii')

    # contract envs with boundary 1, boundary 2
    e_mpo[0]  = np.einsum('wiIx,iIxlLoOy->wlLoOy',boundI_tens,e_mpo[0])    # l(oOr) -- (iIx)lL(oOy)
    e_mpo[-1] = np.einsum('...oOy,zoOy->...z',e_mpo[-1],boundO_tens)       # (iIx)lL(oOy) -- l(iIr)

    if L1 > 1:    # if L1 == 1 don't need to do any reshaping
        e_mpo[0]  = tf.reshape(e_mpo[0], 'i,i,i,iii')
        e_mpo[-1] = tf.reshape(e_mpo[-1],'iii,i,i,i')
        for i in range(1,L1-1):    e_mpo[i] = tf.reshape(e_mpo[i],'iii,i,i,iii')
    
    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,1)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,0,tol=c_tol)

    if len(err) > 0:
        if np.max(err) > 1.0e-1:
            print [np.linalg.norm(m) for m in e_mpo]
            print [np.linalg.norm(m) for m in env_mpo], np.linalg.norm(boundI_tens), np.linalg.norm(boundO_tens)
            print [np.linalg.norm(m) for m in peps_u_col]
            print [np.linalg.norm(m) for m in peps_d_col]
            print err
            raise RuntimeWarning('subR large compression error, XMAX '+str(XMAX))

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
######### ensure envs are Hermitian #########
#############################################

def make_env_hermitian(bound,DMAX=-1,direction=0,use_tens=False):

    if direction == 0:   svd_str = 'ijk,...'
    elif direction == 1: svd_str = '...,ijk'

    if use_tens:
        btens = bound.getSites()

        axT = [0]
        for i in range(len(bound)):  axT += [2*i+2, 2*i+1]
        axT += [btens.ndim-1]

        bH = 0.5*(btens + np.conj(btens.transpose(axT)))
        bH_list = tf.decompose_block(bH,len(bound),direction,DMAX,svd_str=svd_str)
        bound_H = MPX.MPX(bH_list)
    else:
        bound_T = MPX.MPX([np.conj(m.transpose(0,2,1,3)) for m in bound])
        bound_H = MPX.add( MPX.mul(0.5,bound), MPX.mul(0.5, bound_T) )
        bound_H = MPX.canonicalize(bound_H,(direction+1)%2)
        bound_H, errs = MPX.compress_reg(bound_H,DMAX,direction) 

    return bound_H


#############################################
####  apply lambdas to boundary mpos    #####
#############################################


def apply_lam_to_boundary_mpo(b_mpo,bonds_u,bonds_d,op='none'):

    b_new = []
    if   op == 'sqrt_inv':
        for i in range(len(b_mpo)):
            b_new.append( np.einsum('iudo,Uu,Dd->iUDo',b_mpo[i],np.diag(1./np.sqrt(bonds_u[i])),
                                                                np.diag(1./np.sqrt(bonds_d[i])) ) )
    elif op == 'sqrt':
        for i in range(len(b_mpo)):
            b_new.append( np.einsum('iudo,Uu,Dd->iUDo',b_mpo[i],np.diag(np.sqrt(bonds_u[i])),
                                                                np.diag(np.sqrt(bonds_d[i])) ) )
    elif op == 'inv':
        for i in range(len(b_mpo)):
            b_new.append( np.einsum('iudo,Uu,Dd->iUDo',b_mpo[i],np.diag(1./bonds_u[i]),
                                                                np.diag(1./bonds_d[i]) ) )
    elif op == 'full':
        for i in range(len(b_mpo)):
            b_new.append( np.einsum('iudo,Uu,Dd->iUDo',b_mpo[i],np.diag(bonds_u[i]),
                                                                np.diag(bonds_d[i]) ) )
    elif op == 'none':
        b_new = b_mpo

    return b_new


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


def contract_2_bounds(bound1, bound2, bonds_u, bonds_d):
    ''' eg. bL -- bR, bI -- bO '''

    ## lambda are not included on uncontracted bonds
    b2_ = np.einsum('ludr,uU->lUdr',bound2[0],np.diag(bonds_u[0]))
    b2_ = np.einsum('ludr,dD->luDr',b2_,np.diag(bonds_d[0]))
    output = np.einsum('ludr,LudR->lLrR',bound1[0],b2_)
    for m in range(1,len(bound1)):
        b2_ = np.einsum('ludr,uU->lUdr',bound2[m],np.diag(bonds_u[m]))
        b2_ = np.einsum('ludr,dD->luDr',b2_,np.diag(bonds_d[m]))
        # output = np.einsum('lLrR,ruds,RudS->lLsS',output,bound1[m],b2_)
        output = np.einsum('lLrR,ruds->lLRuds',output,bound1[m])
        output = np.einsum('lLRuds,RudS->lLsS',output,b2_)

    # ## lambda are included on uncontracted bonds
    # b2_ = np.einsum('ludr,uU->lUdr',bound2[0],1./bonds_u[0])
    # b2_ = np.einsum('ludr,dD->luDr',b2_,1./bonds_d[0])
    # output = np.einsum('ludr,LudR->lLrR',bound1[0],b2_)
    # for m in range(1,len(bound1)):
    #     b2_ = np.einsum('ludr,uU->lUdr',bound2[m],1./bonds_u[m])
    #     b2_ = np.einsum('ludr,dD->luDr',b2_,1./bonds_d[m])
    #     output = np.einsum('lLrR,ruds,RudS->lLsS',output,bound1[m],b2_)

    
    return np.einsum('llss->',output)



#############################################
#####         contract full env     #########
#############################################

## can also be done from PEPX class, dotting bra+ket and calling PEP0_env contract fct ##


## alternative method (following boundary method more closely ##
def get_ovlp(pepx_u, pepx_d, side='I',XMAX=100,get_err=False):
    ''' contract virtual bonds of pep0 (no physical bonds) '''
   
    L1, L2 = pepx_u.shape

    if   side in ['i','I',1]:   upto = L1
    elif side in ['l','L',0]:   upto = L2
    else:                       upto = 0

    if np.all( [len(dp) == 1 for dp in pepx_u.phys_bonds.flat] ):
        boundaries = get_boundaries(pepx_u,pepx_d,side,upto,XMAX=XMAX,get_err=get_err)
    elif np.all( [len(dp) == 2 for dp in pepx_u.phys_bonds.flat] ):
        # flatten bonds 
        pepx_uf = PEPX.flatten(PEPX.transposeUD(pepx_u))
        pepx_df = PEPX.flatten(PEPX.transposeUD(pepx_d))
        boundaries = get_boundaries(pepx_uf,pepx_df,side,upto,XMAX=XMAX,get_err=get_err)
    else:
        raise TypeError('in env_rho contract:  please use pepo or pep0')
    
    if not get_err:   
        bound = boundaries[upto]

        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        return ovlp
    else:
        bounds, errs = boundaries
        bound = bounds[upto]
       
        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        return ovlp, errs


def get_sub_ovlp(sub_pepx_u, sub_pepx_d, bounds, side=None, XMAX=100, get_err=False,sb_fn=get_subboundaries):

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
            bounds = sb_fn(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bI,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[-1,x,2] for x in range(L2)]
            lam_d = [sub_pepx_d.lambdas[-1,x,2] for x in range(L2)]
            ovlp  = contract_2_bounds(bounds[upto], bO, lam_u, lam_d)
        elif side in ['o','O',2]:
            bounds = sb_fn(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bO,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[0,x,1] for x in range(L2)]
            lam_d = [sub_pepx_d.lambdas[0,x,1] for x in range(L2)]
            ovlp  = contract_2_bounds(bounds[upto], bI, lam_u, lam_d)
        elif side in ['l','L',0]:
            bounds = sb_fn(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bL,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[x,-1,3] for x in range(L1)]
            lam_d = [sub_pepx_d.lambdas[x,-1,3] for x in range(L1)]
            ovlp  = contract_2_bounds(bounds[upto], bR, lam_u, lam_d)
        elif side in ['r','R',3]:
            bounds = sb_fn(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bR,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[x,0,0] for x in range(L1)]
            lam_d = [sub_pepx_d.lambdas[x,0,0] for x in range(L1)]
            ovlp  = contract_2_bounds(bounds[upto], bL, lam_u, lam_d)

        return ovlp

    else:
        if side   in ['i','I',1]:
            bounds,errs = sb_fn(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bI,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[-1,x,2] for x in range(L2)]
            lam_d = [sub_pepx_d.lambdas[-1,x,2] for x in range(L2)]
            ovlp  = contract_2_bounds(bounds[upto], bO, lam_u, lam_d)
        elif side in ['o','O',2]:
            bounds,errs = sb_fn(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bO,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[0,x,1] for x in range(L2)]
            lam_d = [sub_pepx_d.lambdas[0,x,1] for x in range(L2)]
            ovlp  = contract_2_bounds(bounds[upto], bI, lam_u, lam_d)
        elif side in ['l','L',0]:
            bounds,errs = sb_fn(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bL,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[x,-1,3] for x in range(L1)]
            lam_d = [sub_pepx_d.lambdas[x,-1,3] for x in range(L1)]
            ovlp  = contract_2_bounds(bounds[upto], bR, lam_u, lam_d)
        elif side in ['r','R',3]:
            bounds,errs = sb_fn(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bR,XMAX=XMAX,get_errs=get_err)
            lam_u = [sub_pepx_u.lambdas[x,0,0] for x in range(L1)]
            lam_d = [sub_pepx_d.lambdas[x,0,0] for x in range(L1)]
            ovlp  = contract_2_bounds(bounds[upto], bL, lam_u, lam_d)

        return ovlp, errs


#############################################################
####          embedded methods (on the fly env)          ####
#############################################################


# @profile
def embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, idx_key, old_corners=None, return_corners=False):

    # assume sum(shape(bra)) < sum(shape(ket))

    L1,L2 = sub_peps_u.shape
    bL, bI, bO, bR = envs

    # print 'embed sites: lam u', [sub_peps_d.lambdas[idx] for idx in np.ndindex(L1,L2)]

    if (L1,L2) == (1,2):   # horizontal trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0])
            s2d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0])
    
            # env_block2 = np.einsum('wiIx,liord->wlIorxd', bI[1],s2u)
            # env_block2 = np.einsum('wlIorxd,LIORd->wlLoOrRx',env_block2,s2d)        # inner boundary
            # env_block2 = np.einsum('wlLoOrRx,xrRy->wlLoOy',env_block2, bR[0])     # right boundary
            # env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2, bO[1])         # outer boundary
            env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
            env_block2 = np.einsum('wiIrRy,liord->wlIoRdy',env_block2,s2u)
            env_block2 = np.einsum('wlIoRdy,LIORd->wlLoOy',env_block2,s2d)
            env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2,bO[1])
    
            ## site 1 boundary
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0))
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0))

            if idx_key == 'xx':    ## metric
                # env_xx = np.einsum('xiIy,yrRz->xiIrRz',bI[0],env_block2)
                # env_xx = np.einsum('wiIrRz,xoOz->wiIoOrRx',env_xx,bO[0])
                # env_xx = np.einsum('wlLx,wiIoOrRx->lLiIoOrR',bL[0],env_xx)
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,yoOz->wlLiIoOz',env_block1,bO[0])
                out_tens = np.einsum('wlLiIoOz,wrRz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':  ## grad (missing bra)
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,LIORd->wliORdy',env_block1,s1d)
                env_block1 = np.einsum('wliORdy,yoOz->wlioRdz',env_block1,bO[0])
                out_tens = np.einsum('wlioRdz,wrRz->liord',env_block1,env_block2)
            elif idx_key == 'oo':  ## norm
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)
                env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
                env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])
                out_tens = np.einsum('wrRz,wrRz->',env_block1,env_block2)
 
        elif x_idx == (0,1):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[3])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[3])
        
            # env_block1 = np.einsum('xiIw,liord->wlIorxd',bI[0],s1u)
            # env_block1 = np.einsum('wlIorxd,LIORd->wlLoOrRx',env_block1,s1d)     # inner boundary
            # env_block1 = np.einsum('wlLoOrRx,xlLy->woOrRy',env_block1,bL[0])     # left boundary
            # env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])         # outer boundary
            env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)
            env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
            env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])


            ## site 2 boundary
            s2u = PEPX_GL.get_site(sub_peps_u,(0,1))
            s2d = PEPX_GL.get_site(sub_peps_d,(0,1))

            if idx_key == 'xx':
                # env_xx = np.einsum('xiIy,xlLw->wlLiIy',bI[1],env_block1)
                # env_xx = np.einsum('wlLiIy,woOz->lLiIoOyz',env_xx,bO[1])
                # env_xx = np.einsum('lLiIoOyz,yrRz->lLiIoOrR',env_xx,bR[0])
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('wiIrRy,zoOy->wiIoOrRz',env_block2,bO[1])
                out_tens = np.einsum('wlLz,wiIoOrRz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('LIORd,wiIrRy->wLiOrdy',s2d,env_block2)
                env_block2 = np.einsum('wLiOrdy,zoOy->wLiordz',env_block2,bO[1])
                out_tens = np.einsum('wlLz,wLiordz->liord',env_block1,env_block2)
            elif idx_key == 'oo':
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('liord,wiIrRy->wlIoRdy',s2u,env_block2)
                env_block2 = np.einsum('LIORd,wlIoRdy->wlLoOy',s2d,env_block2)
                env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2,bO[1])
                out_tens = np.einsum('wlLz,wlLz->',env_block1,env_block2)

        else:  raise (IndexError)

        if return_corners:      return out_tens, None
        else:                   return out_tens
     

    elif (L1,L2) == (2,1):   # vertical trotter step

        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1])
            s2d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1])
    
            # env_block2 = np.einsum('wlLx,liord->wLiorxd',bL[1],s2u)
            # env_block2 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block2,s2d)     # left boundary
            # env_block2 = np.einsum('xoOy,wiIoOrRx->wiIrRy',bO[0],env_block2)     # outer boundary
            # env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])         # right boundary
            env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
            env_block2 = np.einsum('wlLoOy,liord->wLiOrdy',env_block2,s2u)
            env_block2 = np.einsum('wLiOrdy,LIORd->wiIrRy',env_block2,s2d)
            env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])

            ## site 1 boundary
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0))
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0))
    
            if idx_key == 'xx':
                # env_xx = np.einsum('wlLx,xoOy->wlLoOy',bL[0],env_block2)
                # env_xx = np.einsum('wlLoOy,zrRy->wlLoOrRz',env_xx,bR[0])
                # env_xx = np.einsum('wlLoOrRz,wiIz->lLiIoOrR',env_xx,bI[0])
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,yrRz->wlLiIrRz',env_block1,bR[0])
                out_tens = np.einsum('wlLiIrRz,woOz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,LIORd->wliORdy',env_block1,s1d)
                env_block1 = np.einsum('wliORdy,yrRz->wliOrdz',env_block1,bR[0])
                out_tens = np.einsum('wliOrdz,woOz->liord',env_block1,env_block2)
            elif idx_key == 'oo':
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)
                env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
                env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])
                out_tens = np.einsum('woOz,woOz->',env_block1,env_block2)
    
        elif x_idx == (1,0):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2])
        
            # env_block1 = np.einsum('xlLw,liord->wLiorxd',bL[0],s1u)
            # env_block1 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block1,s1d)     # left boundary
            # env_block1 = np.einsum('xiIy,wiIoOrRx->woOrRy',bI[0],env_block1)     # inner boundary
            # env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])         # right boundary
            env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)   # assume sum(shape(s1u)) < sum(sh(s2u))
            env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
            env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])

            ## site 2 boundary
            s2u = PEPX_GL.get_site(sub_peps_u,(1,0))
            s2d = PEPX_GL.get_site(sub_peps_d,(1,0))
    
            if idx_key == 'xx':
                # env_xx = np.einsum('wlLx,wiIy->xlLiIy',bL[1],env_block1)
                # env_xx = np.einsum('xlLiIy,yrRz->xlLiIrRz',env_xx,bR[1])
                # env_xx = np.einsum('xlLiIrRz,xoOz->lLiIoOrR',env_xx,bO[0])
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,zrRy->wlLoOrRz',env_block2,bR[1])
                out_tens = np.einsum('wiIz,wlLoOrRz->lLiIoOrR',env_block1,env_block2)
            elif idx_key == 'xo':
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,LIORd->wlIoRdy',env_block2,s2d)
                env_block2 = np.einsum('wlIoRdy,zrRy->wlIordz',env_block2,bR[1])
                out_tens = np.einsum('wiIz,wlIordz->liord',env_block1,env_block2)
            elif idx_key == 'oo':
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,liord->wLiOrdy',env_block2,s2u)  # assume sum(shape(s1u)) < sum(sh(s2u)
                env_block2 = np.einsum('wLiOrdy,LIORd->wiIrRy',env_block2,s2d)
                env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])
                out_tens = np.einsum('wiIz,wiIz->',env_block1,env_block2)

        else:  raise (IndexError)

        if return_corners:      return out_tens, None
        else:                   return out_tens
     
   
    elif (L1,L2) == (2,2):  # LR/square trotter step
  
        # @profile
        def get_bL10(bL10x=None,no_lam=[]):
            # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])

            bL1, bO0 = bL[1], bO[0]
            tens10_u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1,3])
            tens10_d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1,3])

            if bL10x is None:
                if np.sum(tens10_u.shape) > np.sum(tens10_d.shape):
                    # bL10 = np.einsum('wlLx,LIORd->wL=lIORdx',bL1,tens10_d)
                    # bL10 = np.einsum('wlIORdx,liord->wiIoOrRx',bL10,tens10_u)
                    # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                    bL10x = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                    bL10x = np.einsum('wlLoOy,LIORd->wlIoRyd',bL10x,tens10_d)
                    bL10x = np.einsum('wlIoRyd,liord->wiIrRy',bL10x,tens10_u)
                else:
                    # bL10 = np.einsum('wlLx,liord->wLiordx',bL1,tens10_u)
                    # bL10 = np.einsum('wLiordx,LIORd->wiIoOrRx',bL10,tens10_d)
                    # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                    bL10x = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                    bL10x = np.einsum('wlLoOy,liord->wLiOryd',bL10x,tens10_u)
                    bL10x = np.einsum('wLiOryd,LIORd->wiIrRy',bL10x,tens10_d)

            bL10 = bL10x
            if 1 not in no_lam:
                bL10 = tf.dot_diag(bL10,sub_peps_u.lambdas[1,0,1],1)
                bL10 = tf.dot_diag(bL10,sub_peps_d.lambdas[1,0,1],2)
            if 3 not in no_lam:
                bL10 = tf.dot_diag(bL10,sub_peps_u.lambdas[1,0,3],3)
                bL10 = tf.dot_diag(bL10,sub_peps_d.lambdas[1,0,3],4)

            return bL10, bL10x
 
        def get_bL11(bL11x=None,no_lam=[]):
            # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])

            bO1, bR1 = bO[1], bR[1]
            tens11_u = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[0,1])
            tens11_d = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[0,1])

            if bL11x is None:
                if np.sum(tens11_u.shape) > np.sum(tens11_d.shape):
                    # bL11 = np.einsum('xoOy,LIORd->xLIoRyd',bO1,tens11_d)
                    # bL11 = np.einsum('xLIoRyd,liord->xlLiIrRy',bL11,tens11_u)
                    # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                    bL11x = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                    bL11x = np.einsum('xoOrRz,LIORd->xLIorzd',bL11x,tens11_d)
                    bL11x = np.einsum('xLIorzd,liord->xlLiIz',bL11x,tens11_u)
                else:
                    # bL11 = np.einsum('xoOy,liord->xliOryd',bO1,tens11_u)
                    # bL11 = np.einsum('xliOryd,LIORd->xlLiIrRy',bL11,tens11_d)
                    # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                    bL11x = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                    bL11x = np.einsum('xoOrRz,liord->xliORzd',bL11x,tens11_u)
                    bL11x = np.einsum('xliORzd,LIORd->xlLiIz',bL11x,tens11_d)

            bL11 = bL11x
            if 0 not in no_lam:
                bL11 = tf.dot_diag(bL11,sub_peps_u.lambdas[1,1,0],1)
                bL11 = tf.dot_diag(bL11,sub_peps_d.lambdas[1,1,0],2)
            if 1 not in no_lam:
                bL11 = tf.dot_diag(bL11,sub_peps_u.lambdas[1,1,1],3)
                bL11 = tf.dot_diag(bL11,sub_peps_d.lambdas[1,1,1],4)

            return bL11, bL11x

        def get_bL01(bL01x=None,no_lam=[]):
            # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])

            bI1, bR0 = bI[1], bR[0]
            tens01_u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0,2])
            tens01_d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0,2])

            if bL01x is None:
                if np.sum(tens01_u.shape) > np.sum(tens01_d.shape):
                    # bL01 = np.einsum('xiIy,LIORd->xLiORyd',bI1,tens01_d)
                    # bL01 = np.einsum('xLiORyd,liord->xlLoOrRy',bL01,tens01_u)
                    # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                    bL01x = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                    bL01x = np.einsum('xiIrRz,LIORd->xLiOrzd',bL01x,tens01_d)
                    bL01x = np.einsum('xLiOrzd,liord->xlLoOz',bL01x,tens01_u)
                else:
                    # bL01 = np.einsum('xiIy,liord->xlIoryd',bI1,tens01_u)
                    # bL01 = np.einsum('xlIoryd,LIORd->xlLoOrRy',bL01,tens01_d)
                    # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                    bL01x = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                    bL01x = np.einsum('xiIrRz,liord->xlIoRzd',bL01x,tens01_u)
                    bL01x = np.einsum('xlIoRzd,LIORd->xlLoOz',bL01x,tens01_d)

            bL01 = bL01x
            if 0 not in no_lam:
                bL01 = tf.dot_diag(bL01,sub_peps_u.lambdas[0,1,0],1)
                bL01 = tf.dot_diag(bL01,sub_peps_d.lambdas[0,1,0],2)
            if 2 not in no_lam:
                bL01 = tf.dot_diag(bL01,sub_peps_u.lambdas[0,1,2],3)
                bL01 = tf.dot_diag(bL01,sub_peps_d.lambdas[0,1,2],4)

            return bL01, bL01x

        def get_bL00(bL00x=None,no_lam=[]):
            # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])

            bI0, bL0 = bI[0], bL[0]
            tens00_u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2,3])
            tens00_d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2,3])

            if bL00x is None:
                if np.sum(tens00_u.shape) > np.sum(tens00_d.shape):
                    # bL00 = np.einsum('ylLx,LIORd->xlIORyd',bL0,tens00_d)
                    # bL00 = np.einsum('xlIORyd,liord->xiIoOrRy',bL00,tens00_u)
                    # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                    bL00x = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                    bL00x = np.einsum('xlLiIz,LIORd->xliORzd',bL00x,tens00_d)
                    bL00x = np.einsum('xliORzd,liord->xoOrRz',bL00x,tens00_u)
                else:
                    # bL00 = np.einsum('ylLx,liord->xLioryd',bL0,tens00_u)
                    # bL00 = np.einsum('xLioryd,LIORd->xiIoOrRy',bL00,tens00_d)
                    # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                    bL00x = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                    bL00x = np.einsum('xlLiIz,liord->xLIorzd',bL00x,tens00_u)
                    bL00x = np.einsum('xLIorzd,LIORd->xoOrRz',bL00x,tens00_d)

            bL00 = bL00x
            if 2 not in no_lam:
                bL00 = tf.dot_diag(bL00,sub_peps_u.lambdas[0,0,2],1)
                bL00 = tf.dot_diag(bL00,sub_peps_d.lambdas[0,0,2],2)
            if 3 not in no_lam:
                bL00 = tf.dot_diag(bL00,sub_peps_u.lambdas[0,0,3],3)
                bL00 = tf.dot_diag(bL00,sub_peps_d.lambdas[0,0,3],4)

            return bL00, bL00x


        ################################
        ###### contractions code #######
        ################################

        if old_corners is not None:           bL00x, bL01x, bL10x, bL11x = old_corners
        else:                                 bL00x, bL01x, bL10x, bL11x = [None,None,None,None]

        ### order contractions assuming 'rol' mpo connectivity ###
        ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
        ###                               10 < 00 ~ 11 < 01 for 3-body operator
        if x_idx == (0,0):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[1,3])
            bL11,bL11x = get_bL11(bL11x,no_lam=[])
            bL01,bL01x = get_bL01(bL01x,no_lam=[0,2])

            ## 10 -> 11 -> 01
            bLs = np.einsum('wiIrRx,xrRoOy->wiIoOy',bL10,bL11)
            bLs = np.einsum('wiIoOy,zlLoOy->wiIlLz',bLs, bL01)

            ## 00 corner 
            su00 = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[])
            sd00 = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[])

            if idx_key == 'xx':    ## metric
                # env_xx = np.einsum('xiIy,woOrRy->xwiIoOrR',bI[0],bLs)
                # env_xx = np.einsum('xlLw,xwiIoOrR->lLiIoOrR',bL[0],env_xx)
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                out_tens = np.einsum('xlLiIz,xoOrRz->lLiIoOrR',bL00,bLs)
            elif idx_key == 'xo':
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                bL00 = np.einsum('xlLiIz,LIORd->xliORdz',bL00,sd00)
                out_tens = np.einsum('xliORdz,xoOrRz->liord',bL00,bLs)
            elif idx_key == 'oo':
                bL00 = get_bL00()[0]   #(bL[0],su00,sd00,bI[0])
                out_tens = np.einsum('xoOrRz,xoOrRz->',bL00,bLs)

            if return_corners:      return out_tens, [None,bL01x,bL10x,bL11x]
            else:                   return out_tens

        elif x_idx == (0,1):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[])
            bL11,bL11x = get_bL11(bL11x,no_lam=[0,1])
            bL00,bL00x = get_bL00(bL00x,no_lam=[2,3])

            ## 00 -> 10 -> 11
            bLs = np.einsum('xoOrRw,xoOlLy->wrRlLy',bL00,bL10)
            bLs = np.einsum('wrRlLy,ylLiIz->wrRiIz',bLs ,bL11)

            ## 01 corner
            su01 = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[])
            sd01 = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[])

            if idx_key == 'xx':    ## metric
                # env_xx = np.einsum('wlLoOy,wiIx->lLiIoOxy',bLs,bI[1])
                # env_xx = np.einsum('lLiIoOxy,xrRy->lLiIoOrR',env_xx,bR[0])
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                out_tens = np.einsum('xiIrRz,xlLoOz->lLiIoOrR',bL01,bLs)
            elif idx_key == 'xo':  ## gradient
                bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                bL01 = np.einsum('xiIrRz,LIORd->xLiOrdz',bL01,sd01)
                out_tens = np.einsum('xlLoOz,xLiOrdz->liord',bLs,bL01)
            else:
                bL01 = get_bL01()[0]  #(bI[1],su01,sd01,bR[0])
                out_tens = np.einsum('xlLoOz,xlLoOz->',bLs,bL01)

            if return_corners:      return out_tens, [bL00x,None,bL10x,bL11x]
            else:                   return out_tens

        elif x_idx == (1,0):

            # update other corners
            bL01,bL01x = get_bL01(bL01x,no_lam=[])
            bL11,bL11x = get_bL11(bL11x,no_lam=[0,1])
            bL00,bL00x = get_bL00(bL00x,no_lam=[2,3])

            ## 00 -> 01 -> 11
            bLs = np.einsum('woOrRx,xrRiIy->woOiIy',bL00,bL01)
            bLs = np.einsum('woOiIy,zlLiIy->woOlLz',bLs, bL11)
 
            ## 10 corner
            su10 = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[])
            sd10 = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[])
 
            if idx_key == 'xx':
                # env_xx = np.einsum('wiIrRz,xoOz->wxiIoOrR',bLs,bO[0])
                # env_xx = np.einsum('wlLx,wxiIoOrR->lLiIoOrR',bL[1],env_xx)
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                out_tens = np.einsum('xlLoOz,xiIrRz->lLiIoOrR',bL10,bLs)
            elif idx_key == 'xo':
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                bL10 = np.einsum('xlLoOz,LIORd->xlIoRdz',bL10,sd10)
                out_tens = np.einsum('xlIoRdz,xiIrRz->liord',bL10,bLs)
            elif idx_key == 'oo':
                bL10 = get_bL10()[0]   #(bL[1],su10,sd10,bO[0])
                out_tens = np.einsum('xiIrRz,xiIrRz->',bL10,bLs)

            if return_corners:      return out_tens, [bL00x,bL01x,None,bL11x]
            else:                   return out_tens

        elif x_idx == (1,1):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[1,3])
            bL00,bL00x = get_bL00(bL00x,no_lam=[])
            bL01,bL01x = get_bL01(bL01x,no_lam=[0,2])

            ## 00 -> 10 -> 01
            bLs = np.einsum('xiIrRw,xiIlLy->wrRlLy',bL10,bL00)
            bLs = np.einsum('wrRlLy,ylLoOz->wrRoOz',bLs, bL01)

            ## 11 corner
            su11 = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[])
            sd11 = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[])

            if idx_key == 'xx':
                # env_xx = np.einsum('wlLiIy,woOx->lLiIoOyx',bLs,bO[1])
                # env_xx = np.einsum('lLiIoOyx,yrRx->lLiIoOrR',env_xx,bR[1])
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                out_tens = np.einsum('xlLiIz,xoOrRz->lLiIoOrR',bLs,bL11)
            elif idx_key == 'xo':
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                bL11 = np.einsum('xoOrRz,LIORd->xLIordz',bL11,sd11)
                out_tens = np.einsum('xlLiIz,xLIordz->liord',bLs,bL11)
            elif idx_key == 'oo':
                bL11 = get_bL11()[0]   #(bO[1],su11,sd11,bR[1])
                out_tens = np.einsum('xlLiIz,xlLiIz->',bLs,bL11)

            if return_corners:      return out_tens, [bL00x,bL01x,bL10x,None]
            else:                   return out_tens

        else:  raise (IndexError)

    else:
        raise (NotImplementedError)
           

def embed_sites_xo(sub_peps_u, sub_peps_d, envs, x_idx, old_corners=None, return_corners=False):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    return embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, 'xo', old_corners=old_corners, return_corners=return_corners)


# @profile
def embed_sites_xx(sub_peps_u, sub_peps_d, envs, x_idx, old_corners=None,return_corners=False):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''
    return embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, 'xx', old_corners=old_corners, return_corners=return_corners)


def embed_sites_norm(sub_pepx,envs_list,old_corners=None):
    ''' get norm of sub_pepx embedded in env '''

    # out =  embed_sites_ovlp( np.conj(PEPX.transposeUD(sub_pepx)), sub_pepx, envs_list, side, XMAX, get_errs)
    out = embed_sites_ovlp( np.conj(sub_pepx), sub_pepx, envs_list, old_corners=old_corners)

    norm = np.sqrt(out)
    if np.isnan(norm):
        print 'embed norm nan', out, norm
        for env in envs_list:  print 'env', [np.linalg.norm(b) for b in env]
        print 'gammas'
        for i in range(sub_pepx.shape[0]):
            print [np.linalg.norm(m) for m in sub_pepx[i,:]]
        print 're(gamma)'
        for i in range(sub_pepx.shape[0]):
            print [np.linalg.norm(np.real(m)) for m in sub_pepx[i,:]]
        print 'sites'
        for idx in np.ndindex(sub_pepx.shape):
            site = PEPX_GL.get_site(sub_pepx,idx)
            print np.linalg.norm(site)
       
        print 'lambdas'
        for idx in np.ndindex(sub_pepx.shape):
            print sub_pepx.lambdas[idx]

        out2 = get_sub_ovlp(np.conj(sub_pepx), sub_pepx, envs_list, side='I')
        out3 = get_sub_ovlp(np.conj(sub_pepx), sub_pepx, envs_list, side='R')
        out4 = get_sub_ovlp(np.conj(sub_pepx), sub_pepx, envs_list, side='O')
        out5 = get_sub_ovlp(np.conj(sub_pepx), sub_pepx, envs_list, side='L')

        print 'embed norm IROL', out,out2,out3,out4,out5
        if np.any( np.abs(np.array([out2,out3,out4,out5])-out) > 0.1 ):
            print 'gammas, lambdas'
            for idx in np.ndindex(sub_pepx.shape):
                print idx, np.linalg.norm(sub_pepx[idx]), sub_pepx.lambdas[idx]


        for idx in np.ndindex(sub_pepx.shape):
            env_xx = embed_sites_xx(np.conj(sub_pepx),sub_pepx,envs_list,idx,
                                            old_corners=None,return_corners=False)
            env_xx_ = np.transpose(env_xx,[0,2,4,6,1,3,5,7])
            sqdim = int(np.sqrt(np.prod(env_xx_.shape)))
            env_sq = env_xx_.reshape(sqdim,sqdim)
            eD,eV = np.linalg.eig(env_sq)
            print 'idx', np.max(eD), np.min(eD)

            # env_xx_H = 0.5*(env_sq + np.conj(env_sq.T))
            # eD,eV = np.linalg.eig(env_xx_H)
            # eD = np.where(eD > 0, eD, 0*eD)
            env_xx_P_ = np.dot(np.conj(eV.T), tf.dMult('DM',eD,eV))
            env_xx_P_ = env_xx_P_.reshape(env_xx_.shape)
            env_xx_P  = np.transpose(env_xx_P_,[0,4,1,5,2,6,3,7])

            s_idx = PEPX_GL.get_site(sub_pepx,idx,no_lam=[])

            norm_X = np.einsum('lLiIoOrR,liord->LIORd',env_xx,np.conj(s_idx))
            norm_X = np.einsum('LIORd,LIORd->',norm_X,s_idx)
            norm_P = np.einsum('lLiIoOrR,liord->LIORd',env_xx_P,np.conj(s_idx))
            norm_P = np.einsum('LIORd,LIORd->',norm_P,s_idx)
            print 'embed not pos, pos env', norm_X, norm_P

        norm_env = build_env(envs_list,ensure_pos=True)

        for idx in np.ndindex(sub_pepx.shape):
            print 'embed', idx, embed_sites(np.conj(sub_pepx),sub_pepx,envs_list,idx,'oo',old_corners=None)

        raise(RuntimeWarning), 'nan norm'

    #     exit()

    # try:  
    #     norm = np.sqrt(out)
    # except(RuntimeWarning):
    #     print out, norm
    #     for env in envs_list:  print 'env', [np.linalg.norm(b) for b in env]
    #     print 'gammas'
    #     for i in range(sub_pepx.shape[0]):
    #         print [np.linalg.norm(m) for m in sub_pepx[i,:]]
    #     print 'lambdas'
    #     for idx in np.ndindex(sub_pepx.shape):
    #         print sub_pepx.lambdas[idx]
        
        
    return np.sqrt(out)


def embed_sites_ovlp(sub_peps_u, sub_peps_d, envs_list,old_corners=None):

    if sub_peps_u.shape == (1,1):
        bL, bI, bO, bR = envs_list
        su00 = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[])
        sd00 = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[])

        norm = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
        norm = np.einsum('xlLiIz,liord->xLIordz',norm,su00)
        norm = np.einsum('xLIordz,LIORd->xoOrRz',norm,sd00)
        norm = np.einsum('xoOrRz,xoOw->wrRz',norm,bO[0])
        norm = np.einsum('wrRz,zrRw->',norm,bR[0])

    else:
        norm = embed_sites(sub_peps_u,sub_peps_d,envs_list,(0,0),'oo',old_corners=old_corners)

    return norm



# ##################################
# #   embedding:  keeping sites    #
# #     with precomputed env       #
# ##################################

def build_env(envs,ensure_pos=False):
    # env MPOs already do NOT have lambdas on open bonds
    return ENV.build_env(envs,ensure_pos)


def embed_sites_xo_env(sub_peps_u, sub_peps_d, env_tens, x_idx):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    L1,L2 = sub_peps_u.shape
    tens_d_idx = PEPX_GL.get_site(sub_peps_d,x_idx)

    env_xx = embed_sites_xx_env(sub_peps_u,sub_peps_d,env_tens,x_idx)
    env_xo = np.einsum('lLiIoOrR,LIORd->liord',env_xx,tens_d_idx)

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
            s2u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0])
            s2d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0])

            env_block2 = np.einsum('lLiIjJoOpPrR,mjprd->lLiIJoOPmRd',env_tens,s2u)
            env_xx = np.einsum('lLiIJoOPmRd,MJPRd->lLiIoOmM',env_block2, s2d)
    
        elif x_idx == (0,1):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[3])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[3])
        
            env_block1 = np.einsum('lLiIjJoOpPrR,liomd->mLIjJOpPrRd',env_tens,s1u)
            env_xx = np.einsum('mLIjJOpPrRd,LIOMd->mMjJpPrR',env_block1, s1d)
    
        else:  raise (IndexError)
        

    elif (L1,L2) == (2,1):   # vertical trotter step
    
        # envblock:  L1-L2-I1-O2-R1-R2

        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1])
            s2d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1])
    
            env_block2 = np.einsum('lLmMiIoOrRsS,mjosd->lLMiIjOrRSd',env_tens,s2u)
            env_xx = np.einsum('lLMiIjOrRSd,MJOSd->lLiIjJrR',env_block2,s2d)
    
        elif x_idx == (1,0):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2])

            env_block1 = np.einsum('lLmMiIoOrRsS,lijrd->LmMIjoORsSd',env_tens,s1u)
            env_xx = np.einsum('LmMIjoORsSd,LIJRd->mMjJoOsS',env_block1,s1d)

        else:  raise (IndexError)
     
   
    elif (L1,L2) == (2,2):  # LR/square trotter step

        # env tens:  'L1-L2 - I1-I2 - O1-O2 - R1-R2'

        if x_idx == (0,0):

            su10 = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1,3])
            sd10 = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1,3])
            su11 = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[])
            sd11 = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[])
            su01 = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0,2])
            sd01 = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0,2])

            env01 = np.einsum('lLmMiIjJoOpPrRsS,njqrd->lLmMiIqJoOpPnRsSd',env_tens,su01)
            env01 = np.einsum('lLmMiIqJoOpPnRsSd,NJQRd->lLmMiIqQoOpPnNsS',env01,sd01)

            env11 = np.einsum('lLmMiIjJoOpPrRsS,njpsd->lLmMiIJoOPrRnSd',env01,su11)
            env11 = np.einsum('lLmMiIJoOPrRnSd,NJPSd->lLmMiIoOrRnN',env11,sd11)

            env_xx = np.einsum('lLmMiIoOrRsS,mjosd->lLMiIjOrRSd',env11,su10)
            env_xx = np.einsum('lLMiIjOrRSd,MJOSd->lLiIjJrR',env_xx,sd10)

        elif x_idx == (0,1):

            su10 = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[])
            sd10 = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[])
            su11 = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[0,1])
            sd11 = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[0,1])
            su00 = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2,3])
            sd00 = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2,3])

            env00 = np.einsum('lLmMiIjJoOpPrRsS,liqtd->tLmMqIjJoOpPrRsSd',env_tens,su00)
            env00 = np.einsum('tLmMqIjJoOpPrRsSd,LIQTd->tTmMqQjJoOpPrRsS',env00,sd00)
 
            env10 = np.einsum('lLmMiIjJoOpPrRsS,miotd->lLtMIjJOpPrRsSd',env00,su10)
            env10 = np.einsum('lLtMIjJOpPrRsSd,MIOTd->lLtTjJpPrRsS',env10,sd10)

            env_xx = np.einsum('lLmMiIoOrRsS,mjosd->lLMiIjOrRSd',env10,su11)
            env_xx = np.einsum('lLMiIjOrRSd,MJOSd->lLiIjJrR',env_xx,sd11)

        elif x_idx == (1,0):

            su01 = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[])
            sd01 = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[])
            su11 = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[0,1])
            sd11 = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[0,1])
            su00 = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2,3])
            sd00 = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2,3])

            env01 = np.einsum('lLmMiIjJoOpPrRsS,njqrd->lLmMiIqJoOpPnRsSd',env_tens,su01)
            env01 = np.einsum('lLmMiIqJoOpPnRsSd,NJQRd->lLmMiIqQoOpPnNsS',env01,sd01)

            env11 = np.einsum('lLmMiIjJoOpPrRsS,njpsd->lLmMiIJoOPrRnSd',env01,su11)
            env11 = np.einsum('lLmMiIJoOPrRnSd,NJPSd->lLmMiIoOrRnN',env11,sd11)

            env_xx = np.einsum('lLmMiIoOrRsS,lijrd->LmMjIoORsSd',env11,su00)
            env_xx = np.einsum('LmMjIoORsSd,LIJRd->mMjJoOsS',env_xx,sd00)

        elif x_idx == (1,1):

            su10 = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1,3])
            sd10 = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1,3])
            su00 = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[])
            sd00 = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[])
            su01 = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0,2])
            sd01 = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0,2])

            env00 = np.einsum('lLmMiIjJoOpPrRsS,liqtd->tLmMqIjJoOpPrRsSd',env_tens,su00)
            env00 = np.einsum('tLmMqIjJoOpPrRsSd,LIQTd->tTmMqQjJoOpPrRsS',env00,sd00)
 
            env10 = np.einsum('lLmMiIjJoOpPrRsS,miotd->lLtMIjJOpPrRsSd',env00,su10)
            env10 = np.einsum('lLtMIjJOpPrRsSd,MIOTd->lLtTjJpPrRsS',env10,sd10)

            env_xx = np.einsum('lLmMiIoOrRsS,lijrd->LmMjIoORsSd',env10,su01)
            env_xx = np.einsum('LmMjIoORsSd,LIJRd->mMjJoOsS',env_xx,sd01)

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

    tens_u_idx = PEPX_GL.get_site(sub_pepx_u,x_idx)

    env_xo = embed_sites_xo_env(sub_pepx_u, sub_pepx_d, env_tens, x_idx)
    ovlp = np.einsum('liord,liord->',env_xo,tens_u_idx)

    return ovlp



#############################################################
####     reduced embedded methods (on the fly env)       ####
#############################################################


def red_embed_sites_xo(sub_peps_u, sub_peps_d, envs, x_idx, iso_leg, qu_idx=None, qd_idx=None, old_corners=None, return_corners=False):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    return red_embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, iso_leg, 'xo', qu_idx, qd_idx, old_corners, return_corners)


# @profile
def red_embed_sites_xx(sub_peps_u, sub_peps_d, envs, x_idx, iso_leg, qu_idx=None, qd_idx=None, old_corners=None, return_corners=False):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''
    return red_embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, iso_leg, 'xx', qu_idx, qd_idx, old_corners, return_corners)


def red_embed_sites(sub_peps_u, sub_peps_d, envs, x_idx, iso_leg, idx_key, qu_idx=None, qd_idx=None, old_corners=None, return_corners=False):

    L1,L2 = sub_peps_u.shape
    bL, bI, bO, bR = envs

    tens_u_idx = PEPX_GL.get_site(sub_peps_u,x_idx)
    tens_d_idx = PEPX_GL.get_site(sub_peps_d,x_idx)

    
    if qu_idx is None or qd_idx is None:
        qu_idx, ru_idx, axT_inv = PEPX.QR_factor(tens_u_idx,iso_leg,d_end=True)
        qd_idx, rd_idx, axT_inv = PEPX.QR_factor(tens_d_idx,iso_leg,d_end=True)
        # print 'red env axT', axT_inv
        return_all = True
    else: 
        return_all = False 


    if (L1,L2) == (1,2):   # horizontal trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0])
            s2d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0])
    
            env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
            env_block2 = np.einsum('wiIrRy,liord->wlIoRdy',env_block2,s2u)
            env_block2 = np.einsum('wlIoRdy,LIORd->wlLoOy',env_block2,s2d)
            env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2,bO[1])
    
            ## site 1 boundary
            # assume iso_leg = 'r'
            if idx_key == 'xx':    ## metric
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lioq->wLIoqy',env_block1,qu_idx)
                env_block1 = np.einsum('wLIoqy,LIOQ->woOqQy',env_block1,qd_idx)
                env_block1 = np.einsum('yoOz,woOqQy->wqQz',bO[0],env_block1)
                env_out = np.einsum('wqQz,wrRz->qQrR',env_block1,env_block2)
            elif idx_key == 'xo':  ## grad (missing bra)
                s1d = PEPX_GL.get_site(sub_peps_d,(0,0))
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lioq->wLIoqy',env_block1,qu_idx)
                env_block1 = np.einsum('wLIoqy,LIORd->woORqdy',env_block1,s1d)
                env_block1 = np.einsum('woORqdy,yoOz->wRqdz',env_block1,bO[0])
                env_out = np.einsum('wRqdz,wrRz->qrd',env_block1,env_block2)
    

        elif x_idx == (0,1):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[3])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[3])
        
            # env_block1 = np.einsum('xiIw,liord->wlIorxd',bI[0],s1u)
            # env_block1 = np.einsum('wlIorxd,LIORd->wlLoOrRx',env_block1,s1d)     # inner boundary
            # env_block1 = np.einsum('wlLoOrRx,xlLy->woOrRy',env_block1,bL[0])     # left boundary
            # env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])         # outer boundary
            env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)
            env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
            env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])

            ## site 2 boundary
            # assume iso_leg = 'l'
            if idx_key == 'xx':
                env_block2 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                env_block2 = np.einsum('xiIrRz,iorq->xqIoRz',env_block2,qu_idx)
                env_block2 = np.einsum('xqIoRz,IORQ->xqQoOz',env_block2,qd_idx)
                env_block2 = np.einsum('xqQoOy,zoOy->xqQz',env_block2,bO[1])
                env_out = np.einsum('wlLz,wqQz->qQlL',env_block1,env_block2)
            elif idx_key == 'xo':
                s2d = PEPX_GL.get_site(sub_peps_d,(0,1))
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('iorq,wiIrRy->wqIoRy',qu_idx,env_block2)
                env_block2 = np.einsum('LIORd,wqIoRy->wqLoOdy',s2d,env_block2)
                env_block2 = np.einsum('wqLoOdy,zoOy->wqLdz',env_block2,bO[1])
                env_out = np.einsum('wlLz,wqLdz->qld',env_block1,env_block2)

        else:  raise (IndexError)

        if return_all:
            if return_corners:     return env_out, qu_idx, qd_idx, axT_inv, None
            else:                  return env_out, qu_idx, qd_idx, axT_inv
        else:
            if return_corners:     return env_out, None
            else:                  return env_out
        

    elif (L1,L2) == (2,1):   # vertical trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1])
            s2d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1])
    
            # env_block2 = np.einsum('wlLx,liord->wLiorxd',bL[1],s2u)
            # env_block2 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block2,s2d)     # left boundary
            # env_block2 = np.einsum('xoOy,wiIoOrRx->wiIrRy',bO[0],env_block2)     # outer boundary
            # env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])         # right boundary
            env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
            env_block2 = np.einsum('wlLoOy,liord->wLiOrdy',env_block2,s2u)
            env_block2 = np.einsum('wLiOrdy,LIORd->wiIrRy',env_block2,s2d)
            env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])

            ## site 1 boundary
            # assume iso_leg = O
            if idx_key == 'xx':
                env_block1 = np.einsum('wlLx,wiIy->xlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('xlLiIy,lirq->xLIqry',env_block1,qd_idx)
                env_block1 = np.einsum('xLIqry,LIRQ->xqQrRy',env_block1,qu_idx)
                env_block1 = np.einsum('xqQrRy,yrRz->xqQz',env_block1,bR[0])
                env_out = np.einsum('xqQz,xoOz->qQoO',env_block1,env_block2)
            elif idx_key == 'xo':
                s1d = PEPX_GL.get_site(sub_peps_d,(0,0))
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lirq->wLIqry',env_block1,qu_idx)
                env_block1 = np.einsum('wLIqry,LIORd->wqOrRdy',env_block1,s1d)
                env_block1 = np.einsum('wqOrRdy,yrRz->wqOdz',env_block1,bR[0])
                env_out = np.einsum('wqOdz,woOz->qod',env_block1,env_block2)


        elif x_idx == (1,0):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2])
        
            # env_block1 = np.einsum('xlLw,liord->wLiorxd',bL[0],s1u)
            # env_block1 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block1,s1d)     # left boundary
            # env_block1 = np.einsum('xiIy,wiIoOrRx->woOrRy',bI[0],env_block1)     # inner boundary
            # env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])         # right boundary
            env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)   # assume sum(shape(s1u)) < sum(sh(s2u))
            env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
            env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])

            ## site 2 boundary
            # assume iso_leg = I
    
            if idx_key == 'xx':
                env_block2 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                env_block2 = np.einsum('xlLoOz,lorq->xLqOrz',env_block2,qu_idx)
                env_block2 = np.einsum('xLqOrz,LORQ->xqQrRz',env_block2,qd_idx)
                env_block2 = np.einsum('xqQrRz,yrRz->xqQy',env_block2,bR[1])
                env_out = np.einsum('xiIz,xqQz->qQiI',env_block1,env_block2)
            elif idx_key == 'xo':
                s2d = PEPX_GL.get_site(sub_peps_d,(1,0))
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,lorq->wLqOry',env_block2,qu_idx)
                env_block2 = np.einsum('wLqOry,LIORd->wqIrRdy',env_block2,s2d)
                env_block2 = np.einsum('wqIrRdy,zrRy->wqIdz',env_block2,bR[1])
                env_out = np.einsum('wiIz,wqIdz->qid',env_block1,env_block2)

        else:  raise (IndexError)

        if return_all:
            if return_corners:     return env_out, qu_idx, qd_idx, axT_inv, None
            else:                  return env_out, qu_idx, qd_idx, axT_inv
        else:
            if return_corners:     return env_out, None
            else:                  return env_out
   

    elif (L1,L2) == (2,2):  # LR/square trotter step

        # def get_bL10(bL1,tens10_u,tens10_d,bO0):
        #     if np.sum(tens10_u.shape) > np.sum(tens10_d.shape):
        #         bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        #         bL10 = np.einsum('wlLoOy,LIORd->wlIoRyd',bL10,tens10_d)
        #         bL10 = np.einsum('wlIoRyd,liord->wiIrRy',bL10,tens10_u)
        #     else:
        #         bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        #         bL10 = np.einsum('wlLoOy,liord->wLiOryd',bL10,tens10_u)
        #         bL10 = np.einsum('wLiOryd,LIORd->wiIrRy',bL10,tens10_d)
        #     return bL10
 
        # def get_bL11(bO1,tens11_u,tens11_d,bR1):
        #     if np.sum(tens11_u.shape) > np.sum(tens11_d.shape):
        #         bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        #         bL11 = np.einsum('xoOrRz,LIORd->xLIorzd',bL11,tens11_d)
        #         bL11 = np.einsum('xLIorzd,liord->xlLiIz',bL11,tens11_u)
        #     else:
        #         bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        #         bL11 = np.einsum('xoOrRz,liord->xliORzd',bL11,tens11_u)
        #         bL11 = np.einsum('xliORzd,LIORd->xlLiIz',bL11,tens11_d)
        #     return bL11

        # def get_bL01(bI1,tens01_u,tens01_d,bR0):
        #     if np.sum(tens01_u.shape) > np.sum(tens01_d.shape):
        #         bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        #         bL01 = np.einsum('xiIrRz,LIORd->xLiOrzd',bL01,tens01_d)
        #         bL01 = np.einsum('xLiOrzd,liord->xlLoOz',bL01,tens01_u)
        #     else:
        #         bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        #         bL01 = np.einsum('xiIrRz,liord->xlIoRzd',bL01,tens01_u)
        #         bL01 = np.einsum('xlIoRzd,LIORd->xlLoOz',bL01,tens01_d)
        #     return bL01

        # def get_bL00(bL0,tens00_u,tens00_d,bI0):
        #     if np.sum(tens00_u.shape) > np.sum(tens00_d.shape):
        #         bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        #         bL00 = np.einsum('xlLiIz,LIORd->xliORzd',bL00,tens00_d)
        #         bL00 = np.einsum('xliORzd,liord->xoOrRz',bL00,tens00_u)
        #     else:
        #         bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        #         bL00 = np.einsum('xlLiIz,liord->xLIorzd',bL00,tens00_u)
        #         bL00 = np.einsum('xLIorzd,LIORd->xoOrRz',bL00,tens00_d)
        #     return bL00

        def get_bL10(bL10x=None,no_lam=[]):
            # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])

            bL1, bO0 = bL[1], bO[0]
            tens10_u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1,3])
            tens10_d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1,3])

            if bL10x is None:
                if np.sum(tens10_u.shape) > np.sum(tens10_d.shape):
                    # bL10 = np.einsum('wlLx,LIORd->wL=lIORdx',bL1,tens10_d)
                    # bL10 = np.einsum('wlIORdx,liord->wiIoOrRx',bL10,tens10_u)
                    # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                    bL10x = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                    bL10x = np.einsum('wlLoOy,LIORd->wlIoRyd',bL10x,tens10_d)
                    bL10x = np.einsum('wlIoRyd,liord->wiIrRy',bL10x,tens10_u)
                else:
                    # bL10 = np.einsum('wlLx,liord->wLiordx',bL1,tens10_u)
                    # bL10 = np.einsum('wLiordx,LIORd->wiIoOrRx',bL10,tens10_d)
                    # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                    bL10x = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                    bL10x = np.einsum('wlLoOy,liord->wLiOryd',bL10x,tens10_u)
                    bL10x = np.einsum('wLiOryd,LIORd->wiIrRy',bL10x,tens10_d)

            bL10 = bL10x
            if 1 not in no_lam:
                bL10 = tf.dot_diag(bL10,sub_peps_u.lambdas[1,0,1],1)
                bL10 = tf.dot_diag(bL10,sub_peps_d.lambdas[1,0,1],2)
            if 3 not in no_lam:
                bL10 = tf.dot_diag(bL10,sub_peps_u.lambdas[1,0,3],3)
                bL10 = tf.dot_diag(bL10,sub_peps_d.lambdas[1,0,3],4)

            return bL10, bL10x
 
        def get_bL11(bL11x=None,no_lam=[]):
            # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])

            bO1, bR1 = bO[1], bR[1]
            tens11_u = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[0,1])
            tens11_d = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[0,1])

            if bL11x is None:
                if np.sum(tens11_u.shape) > np.sum(tens11_d.shape):
                    # bL11 = np.einsum('xoOy,LIORd->xLIoRyd',bO1,tens11_d)
                    # bL11 = np.einsum('xLIoRyd,liord->xlLiIrRy',bL11,tens11_u)
                    # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                    bL11x = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                    bL11x = np.einsum('xoOrRz,LIORd->xLIorzd',bL11x,tens11_d)
                    bL11x = np.einsum('xLIorzd,liord->xlLiIz',bL11x,tens11_u)
                else:
                    # bL11 = np.einsum('xoOy,liord->xliOryd',bO1,tens11_u)
                    # bL11 = np.einsum('xliOryd,LIORd->xlLiIrRy',bL11,tens11_d)
                    # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                    bL11x = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                    bL11x = np.einsum('xoOrRz,liord->xliORzd',bL11x,tens11_u)
                    bL11x = np.einsum('xliORzd,LIORd->xlLiIz',bL11x,tens11_d)

            bL11 = bL11x
            if 0 not in no_lam:
                bL11 = tf.dot_diag(bL11,sub_peps_u.lambdas[1,1,0],1)
                bL11 = tf.dot_diag(bL11,sub_peps_d.lambdas[1,1,0],2)
            if 1 not in no_lam:
                bL11 = tf.dot_diag(bL11,sub_peps_u.lambdas[1,1,1],3)
                bL11 = tf.dot_diag(bL11,sub_peps_d.lambdas[1,1,1],4)

            return bL11, bL11x

        def get_bL01(bL01x=None,no_lam=[]):
            # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])

            bI1, bR0 = bI[1], bR[0]
            tens01_u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0,2])
            tens01_d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0,2])

            if bL01x is None:
                if np.sum(tens01_u.shape) > np.sum(tens01_d.shape):
                    # bL01 = np.einsum('xiIy,LIORd->xLiORyd',bI1,tens01_d)
                    # bL01 = np.einsum('xLiORyd,liord->xlLoOrRy',bL01,tens01_u)
                    # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                    bL01x = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                    bL01x = np.einsum('xiIrRz,LIORd->xLiOrzd',bL01x,tens01_d)
                    bL01x = np.einsum('xLiOrzd,liord->xlLoOz',bL01x,tens01_u)
                else:
                    # bL01 = np.einsum('xiIy,liord->xlIoryd',bI1,tens01_u)
                    # bL01 = np.einsum('xlIoryd,LIORd->xlLoOrRy',bL01,tens01_d)
                    # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                    bL01x = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                    bL01x = np.einsum('xiIrRz,liord->xlIoRzd',bL01x,tens01_u)
                    bL01x = np.einsum('xlIoRzd,LIORd->xlLoOz',bL01x,tens01_d)

            bL01 = bL01x
            if 0 not in no_lam:
                bL01 = tf.dot_diag(bL01,sub_peps_u.lambdas[0,1,0],1)
                bL01 = tf.dot_diag(bL01,sub_peps_d.lambdas[0,1,0],2)
            if 2 not in no_lam:
                bL01 = tf.dot_diag(bL01,sub_peps_u.lambdas[0,1,2],3)
                bL01 = tf.dot_diag(bL01,sub_peps_d.lambdas[0,1,2],4)

            return bL01, bL01x

        def get_bL00(bL00x=None,no_lam=[]):
            # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])

            bI0, bL0 = bI[0], bL[0]
            tens00_u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2,3])
            tens00_d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2,3])

            if bL00x is None:
                if np.sum(tens00_u.shape) > np.sum(tens00_d.shape):
                    # bL00 = np.einsum('ylLx,LIORd->xlIORyd',bL0,tens00_d)
                    # bL00 = np.einsum('xlIORyd,liord->xiIoOrRy',bL00,tens00_u)
                    # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                    bL00x = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                    bL00x = np.einsum('xlLiIz,LIORd->xliORzd',bL00x,tens00_d)
                    bL00x = np.einsum('xliORzd,liord->xoOrRz',bL00x,tens00_u)
                else:
                    # bL00 = np.einsum('ylLx,liord->xLioryd',bL0,tens00_u)
                    # bL00 = np.einsum('xLioryd,LIORd->xiIoOrRy',bL00,tens00_d)
                    # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                    bL00x = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                    bL00x = np.einsum('xlLiIz,liord->xLIorzd',bL00x,tens00_u)
                    bL00x = np.einsum('xLIorzd,LIORd->xoOrRz',bL00x,tens00_d)

            bL00 = bL00x
            if 2 not in no_lam:
                bL00 = tf.dot_diag(bL00,sub_peps_u.lambdas[0,0,2],1)
                bL00 = tf.dot_diag(bL00,sub_peps_d.lambdas[0,0,2],2)
            if 3 not in no_lam:
                bL00 = tf.dot_diag(bL00,sub_peps_u.lambdas[0,0,3],3)
                bL00 = tf.dot_diag(bL00,sub_peps_d.lambdas[0,0,3],4)

            return bL00, bL00x


        if old_corners is not None:           bL00x, bL01x, bL10x, bL11x = old_corners
        else:                                 bL00x, bL01x, bL10x, bL11x = [None,None,None,None]

        ### order contractions assuming 'rol' mpo connectivity ###
        ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
        ###                               10 < 00 ~ 11 < 01 for 3-body operator
        if x_idx == (0,0):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[1,3])
            bL11,bL11x = get_bL11(bL11x,no_lam=[])
            bL01,bL01x = get_bL01(bL01x,no_lam=[0,2])

            # bL10_,bL10x_ = get_bL10(None,no_lam=[1,3])
            # bL11_,bL11x_ = get_bL11(None,no_lam=[])
            # bL01_,bL01x_ = get_bL01(None,no_lam=[0,2])

            # err = [np.linalg.norm(bL10-bL10_),np.linalg.norm(bL11-bL11_),np.linalg.norm(bL01-bL01_)]
            # if np.any(np.array(err)>1.0e-10):                print '(0,0)',err, [(c is None) for c in old_corners]

            ## 10 -> 11 -> 01
            bLs = np.einsum('wiIrRx,xrRoOy->wiIoOy',bL10,bL11)
            bLs = np.einsum('wiIoOy,zlLoOy->wiIlLz',bLs, bL01)

            ## 00 corner 
            if idx_key == 'xx':    ## metric
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                if   iso_leg in ['r','R',3]:
                    bL00 = np.einsum('xlLiIz,lioq->xLIoqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIoqz,LIOQ->xoOqQz',bL00,qd_idx)
                    env_out = np.einsum('woOqQz,woOrRz->qQrR',bL00,bLs)
                elif iso_leg in ['o','O',2]:
                    bL00 = np.einsum('xlLiIz,lirq->xLIqrz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqrz,LIRQ->xqQrRz',bL00,qd_idx)
                    env_out = np.einsum('wqQrRz,woOrRz->qQoO',bL00,bLs)
                elif iso_leg in ['or','ro','OR','RO']:
                    bL00 = np.einsum('xlLiIz,liq->xLIqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqz,LIQ->xqQz',bL00,qd_idx)
                    env_out = np.einsum('wqQz,woOrRz->qQoOrR',bL00,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':
                sd00 = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[])
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                if   iso_leg in ['r','R',3]:
                    bL00 = np.einsum('xlLiIz,lioq->xLIoqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIoqz,LIORd->xoOqRdz',bL00,sd00)
                    env_out = np.einsum('xoOqRdz,xoOrRz->qrd',bL00,bLs)
                elif iso_leg in ['o','O',2]:
                    bL00 = np.einsum('xlLiIz,lirq->xLIqrz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqrz,LIORd->xqOrRdz',bL00,sd00)
                    env_out = np.einsum('xqOrRdz,xoOrRz->qod',bL00,bLs)
                elif iso_leg in ['or','ro','OR','RO']:
                    bL00 = np.einsum('xlLiIz,liq->xLIqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqz,LIORd->xqORdz',bL00,sd00)
                    env_out = np.einsum('wqORdz,woOrRz->qord',bL00,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [None,bL01x,bL10x,bL11x]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [None,bL01x,bL10x,bL11x]
                else:                 return env_out

        elif x_idx == (0,1):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[])
            bL11,bL11x = get_bL11(bL11x,no_lam=[0,1])
            bL00,bL00x = get_bL00(bL00x,no_lam=[2,3])

            # bL10_,bL10x_ = get_bL10(None,no_lam=[])
            # bL11_,bL11x_ = get_bL11(None,no_lam=[0,1])
            # bL00_,bL00x_ = get_bL00(None,no_lam=[2,3])

            # err = [np.linalg.norm(bL10-bL10_),np.linalg.norm(bL11-bL11_),np.linalg.norm(bL00-bL00_)]
            # if np.any(np.array(err)>1.0e-10):                print '(0,1)', err, [(c is None) for c in old_corners]

            ## 00 -> 10 -> 11
            bLs = np.einsum('xoOrRw,xoOlLy->wrRlLy',bL00,bL10)
            bLs = np.einsum('wrRlLy,ylLiIz->wrRiIz',bLs ,bL11)

            ## 01 corner
            if idx_key == 'xx':    ## metric
                bL01 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                if   iso_leg in ['l','L',0]:
                    bL01 = np.einsum('xiIrRz,iorq->xqIoRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIoRz,IORQ->xqQoOz',bL01,qd_idx)
                    env_out = np.einsum('xqQoOz,xlLoOz->qQlL',bL01,bLs)
                elif iso_leg in ['o','O',2]:
                    bL01 = np.einsum('xiIrRz,lirq->xlIqRz',bL01,qu_idx)
                    bL01 = np.einsum('xlIqRz,LIRQ->xlLqQz',bL01,qd_idx)
                    env_out = np.einsum('xlLqQz,xlLoOz->qQoO',bL01,bLs)
                elif iso_leg in ['lo','LO','ol','OL']:
                    bL01 = np.einsum('xiIrRz,irq->xqIRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIRz,IRQ->xqQz',bL01,qd_idx)
                    env_out = np.einsum('xqQz,xlLoOz->qQlLoO',bL01,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':  ## gradient
                sd01 = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[])
                bL01 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                if   iso_leg in ['l','L',0]:
                    bL01 = np.einsum('xiIrRz,iorq->xqIoRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIoRz,LIORd->xqLoOdz',bL01,sd01)
                    env_out = np.einsum('xqLoOdz,xlLoOz->qld',bL01,bLs)
                elif iso_leg in ['o','O',2]:
                    bL01 = np.einsum('xiIrRz,lirq->xlIqRz',bL01,qu_idx)
                    bL01 = np.einsum('xlIqRz,LIORd->xlLqOdz',bL01,sd01)
                    env_out = np.einsum('xlLqOdz,xlLoOz->qod',bL01,bLs)
                elif iso_leg in ['lo','LO','ol','OL']:
                    bL01 = np.einsum('xiIrRz,irq->xqIRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIRz,LIORd->xqLOdz',bL01,sd01)
                    env_out = np.einsum('xqLOdz,xlLoOz->qlod',bL01,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [bL00x,None,bL10x,bL11x]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [bL00x,None,bL10x,bL11x]
                else:                 return env_out

        elif x_idx == (1,0):

            # update other corners
            bL01,bL01x = get_bL01(bL01x,no_lam=[])
            bL11,bL11x = get_bL11(bL11x,no_lam=[0,1])
            bL00,bL00x = get_bL00(bL00x,no_lam=[2,3])

            # bL01_,bL01x_ = get_bL01(None,no_lam=[])
            # bL11_,bL11x_ = get_bL11(None,no_lam=[0,1])
            # bL00_,bL00x_ = get_bL00(None,no_lam=[2,3])

            # err = [np.linalg.norm(bL00-bL00_),np.linalg.norm(bL11-bL11_),np.linalg.norm(bL01-bL01_)]
            # if np.any(np.array(err)>1.0e-10):                print '(1,0)',err, [(c is None) for c in old_corners]

            ## 00 -> 01 -> 11
            bLs = np.einsum('woOrRx,xrRiIy->woOiIy',bL00,bL01)
            bLs = np.einsum('woOiIy,zlLiIy->woOlLz',bLs, bL11)
 
            ## 10 corner
            if idx_key == 'xx':
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                if   iso_leg in ['i','I',1]:
                    bL10 = np.einsum('xlLoOz,lorq->xLqOrz',bL10,qu_idx)
                    bL10 = np.einsum('xLqOrz,LORQ->xqQrRz',bL10,qd_idx)
                    env_out = np.einsum('xqQrRz,xiIrRz->qQiI',bL10,bLs)
                elif iso_leg in ['r','R',3]:
                    bL10 = np.einsum('xlLoOz,lioq->xLiOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLiOqz,LIOQ->xiIqQz',bL10,qd_idx)
                    env_out = np.einsum('xiIqQz,xiIrRz->qQrR',bL10,bLs)
                elif iso_leg in ['ri','ir','RI','IR']:
                    bL10 = np.einsum('xlLoOz,loq->xLOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLOqz,LOQ->xqQz',bL10,qd_idx)
                    env_out = np.einsum('xqQz,xiIrRz->qQiIrR',bL10,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':
                sd10 = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[])
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                if   iso_leg in ['i','I',1]:
                    bL10 = np.einsum('xlLoOz,lorq->xLqOrz',bL10,qu_idx)
                    bL10 = np.einsum('xLqOrz,LIORd->xqIrRdz',bL10,sd10)
                    env_out = np.einsum('xqIrRdz,xiIrRz->qid',bL10,bLs)
                elif iso_leg in ['r','R',3]:
                    bL10 = np.einsum('xlLoOz,lioq->xLiOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLiOqz,LIORd->xiIqRdz',bL10,sd10)
                    env_out = np.einsum('xiIqRdz,xiIrRz->qrd',bL10,bLs)
                elif iso_leg in ['ri','ir','RI','IR']:
                    bL10 = np.einsum('xlLoOz,loq->xLOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLOqz,LIORd->xqIRdz',bL10,sd10)
                    env_out = np.einsum('xqIRdz,xiIrRz->qird',bL10,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [bL00x,bL01x,None,bL11x]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [bL00x,bL01x,None,bL11x]
                else:                 return env_out


        elif x_idx == (1,1):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[1,3])
            bL00,bL00x = get_bL00(bL00x,no_lam=[])
            bL01,bL01x = get_bL01(bL01x,no_lam=[0,2])

            ## 00 -> 10 -> 01
            bLs = np.einsum('xiIrRw,xiIlLy->wrRlLy',bL10,bL00)
            bLs = np.einsum('wrRlLy,ylLoOz->wrRoOz',bLs, bL01)

            ## 11 corner
            if idx_key == 'xx':
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                if   iso_leg in ['l','L',0]:
                    bL11 = np.einsum('xoOrRz,iorq->xqiORz',bL11,qu_idx)
                    bL11 = np.einsum('xqiORz,IORQ->xqQiIz',bL11,qd_idx)
                    env_out = np.einsum('xqQiIz,xlLiIz->qQlL',bL11,bLs)
                elif iso_leg in ['i','I',1]:
                    bL11 = np.einsum('xoOrRz,lorq->xlqORz',bL11,qu_idx)
                    bL11 = np.einsum('xlqORz,LORQ->xlLqQz',bL11,qd_idx)
                    env_out = np.einsum('xlLqQz,xlLiIz->qQiI',bL11,bLs)
                elif iso_leg in ['li','il','LI','IL']:
                    bL11 = np.einsum('xoOrRz,orq->xqORz',bL11,qu_idx)
                    bL11 = np.einsum('xqORz,ORQ->xqQz',bL11,qd_idx)
                    env_out = np.einsum('xqQz,xlLiIz->qQlLiI',bL11,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':
                sd11 = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[])
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                if   iso_leg in ['l','L',0]:
                    bL11 = np.einsum('xoOrRz,iorq->xqiORz',bL11,qu_idx)
                    bL11 = np.einsum('xqiORz,LIORd->xqLiIdz',bL11,sd11)
                    env_out = np.einsum('xqLiIdz,xlLiIz->qld',bL11,bLs)
                elif iso_leg in ['i','I',1]:
                    bL11 = np.einsum('xoOrRz,lorq->xlqORz',bL11,qu_idx)
                    bL11 = np.einsum('xlqORz,LIORd->xlLqIdz',bL11,sd11)
                    env_out = np.einsum('xlLqIdz,xlLiIz->qid',bL11,bLs)
                elif iso_leg in ['li','il','LI','IL']:
                    bL11 = np.einsum('xoOrRz,orq->xqORz',bL11,qu_idx)
                    bL11 = np.einsum('xqORz,LIORd->xqLIdz',bL11,sd11)
                    env_out = np.einsum('xqLIdz,xlLiIz->qlid',bL11,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [bL00x,bL01x,bL10x,None]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [bL00x,bL01x,bL10x,None]
                else:                 return env_out

        else:  raise (IndexError)

    else:
        raise (NotImplementedError)
     

#############################################################
####     reduced embedded methods (set env)              ####
#############################################################

def build_env_qr(env_list,qs_u,qs_d,ensure_pos=False):
    ''' qs = list [(0,0),(0,1)] or [(0,0),(1,0)] or [(0,0),(0,1),(1,0),(1,1)]
             None if qr decomp not done at that site
    '''

    bL,bI,bO,bR = env_list
    L1 = len(bL)
    L2 = len(bI)

    if (L1,L2) == (1,2):

        # (0,0) env
        if qs_u[0] is None:
            env0 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
            env0 = np.einsum('wlLiIy,yoOz->wlLiIoOz',env0,bO[0])
        else:
            env0 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
            env0 = np.einsum('wlLiIy,lioq->wLIoqy',env0,np.conj(qs_u[0]))
            env0 = np.einsum('wLIoqy,LIOQ->woOqQy',env0,qs_d[0])
            env0 = np.einsum('woOqQy,yoOz->wqQz',env0,bO[0])

        # (0,1) env
        if qs_u[1] is None:
            env1 = np.einsum('zoOy,xrRy->xoOrRz',bO[1],bR[0])
            env1 = np.einsum('xoOrRz,wiIx->wiIoOrRz',env1,bI[1])
        else:
            env1 = np.einsum('zoOy,xrRy->xoOrRz',bO[1],bR[0])
            env1 = np.einsum('xoOrRz,rioq->xiORqz',env1,np.conj(qs_u[1]))
            env1 = np.einsum('xiORqz,RIOQ->xiIqQz',env1,qs_d[1])
            env1 = np.einsum('xiIqQz,wiIx->wqQz',env1,bI[1])

        env = np.tensordot(env0,env1,axes=((0,-1),(0,-1)))

        if (qs_u[0] is None) and (qs_u[1] is None):
            env = np.einsum('lLiIoOjJpPrR->lLiIjJoOpPrR',env)

    elif (L1,L2) == (2,1):

        # (0,0) env
        if qs_u[0] is None:
            env0 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
            env0 = np.einsum('wlLiIy,yrRz->wlLiIrRz',env0,bR[0])
        else:
            env0 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
            env0 = np.einsum('wlLiIy,ilrq->wLIrqy',env0,np.conj(qs_u[0]))   # Q = in x ... x q 
            env0 = np.einsum('wLIrqy,ILRQ->wrRqQy',env0,qs_d[0])
            env0 = np.einsum('wrRqQy,yrRz->wqQz',env0,bR[0])

        # (0,1) env
        if qs_u[1] is None:
            env1 = np.einsum('xoOy,zrRy->xoOrRz',bO[0],bR[1])
            env1 = np.einsum('xoOrRz,wlLx->wlLoOrRz',env1,bL[1])
        else:
            env1 = np.einsum('xoOy,zrRy->xoOrRz',bO[0],bR[1])
            env1 = np.einsum('xoOrRz,olrq->xlORqz',env1,np.conj(qs_u[1]))
            env1 = np.einsum('xlORqz,OLRQ->xlLqQz',env1,qs_d[1])
            env1 = np.einsum('xlLqQz,wlLx->wqQz',env1,bL[1])

        env = np.tensordot(env0,env1,axes=((0,-1),(0,-1)))

        if (qs_u[0] is None) and (qs_u[1] is None):
            env = np.einsum('lLiIrRmMoOsS->lLmMiIoOrRsS',env)

    elif (L1,L2) == (2,2):        raise(NotImplementedError)
    else:                            raise(NotImplementedError)

    if ensure_pos:

        # print 'build env qr', ensure_pos

        sqdim = int(np.sqrt(np.prod(env.shape)))
        axT = np.arange(env.ndim).reshape(-1,2).T.reshape(-1)
        axT_inv = np.argsort(axT)

        env_ = env.transpose(axT)
        env_sq = env_.reshape(sqdim,sqdim)

        # ## Reza's method (sqrt(M^* M)
        # u,s,vt = np.linalg.svd(env_sq)
        # env_pos_sq = np.dot(np.conj(vt.T),tf.dMult('DM',s,vt))
        # env_pos = np.transpose(env_pos_sq.reshape(env_.shape),axT_inv)
        
        ## Lubasch's method:  (M + M^*)/2 -> pos eigvals
        env_temp = 1./2 * (env_sq + np.conj(env_sq.T))
        evals, evecs = np.linalg.eigh(env_temp)
        evals_pos = np.where(evals>0,evals,np.zeros(evals.shape))
        env_pos_sq = np.dot(tf.dMult('MD',evecs,evals_pos),np.conj(evecs.T))
        env_pos = np.transpose(env_pos_sq.reshape(env_.shape),axT_inv)

        # print 'pos env diff norm', np.linalg.norm(env-env_pos)
        env = env_pos

    return env


def red_embed_sites_xo_env(Ls, gam_u_dict, lam_u, gam_d_dict, lam_d, env, x_idx):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    return red_embed_sites_env(Ls, gam_u_dict, lam_u, gam_d_dict, lam_d, env, x_idx, 'xo')


def red_embed_sites_xx_env(Ls, gam_u_dict, lam_u, gam_d_dict, lam_d, env, x_idx):
    ''' returns embedded system with site x_idx missing in both bra+ket
        for 1x2 or 2x1 system; lam is lam between two sites in gam_dict (both qdx)
    '''
    return red_embed_sites_env(Ls, gam_u_dict, lam_u, gam_d_dict, lam_d, env, x_idx, 'xx')


def red_embed_sites_norm_env(Ls, gam_dict, lam, env, x_idx=(0,0)):
    ovlp = red_embed_sites_env(Ls, gam_dict, lam, gam_dict, lam, env, x_idx, 'oo')
    return np.sqrt(ovlp)


def red_embed_sites_ovlp_env(Ls, gam_u_dict, lam_u, gam_d_dict, lam_d, env, x_idx=(0,0)):
    return red_embed_sites_env(Ls, gam_u_dict, lam_u, gam_d_dict, lam_d, env, x_idx, 'oo')


def red_embed_sites_env(Ls, gam_u_dict, lam_u, gam_d_dict, lam_d, env, x_idx, idx_key):
    # different from above, need to take care of conj in fct
    # qr:  tens_dict tens axes = qdr 

    if Ls == (1,2):

        if x_idx == (0,0): 

            # print 'red embed env', env.shape, tens_u_dict[0,0].shape, tens_u_dict[0,1].shape
            env_out = np.einsum('qQvV,vdl->qQVdl',env,np.conj(gam_u_dict[0,1]))
            env_out = np.einsum('qQVdl,VdL->qQlL',env_out,gam_d_dict[0,1])           # metric

            if idx_key == 'xo' or idx_key == 'oo':
                tens00_d = tf.dMult('MD',gam_d_dict[0,0],lam_d)    # qdx * lam(x)
                env_out = np.einsum('qQrR,QdR->qrd', env_out,tens00_d)        # gradient
                ## need to check order of ops
            if idx_key == 'oo':
                tens00_u = tf.dMult('MD',gam_u_dict[0,0],lam_u)    # qdx * lam(x)
                env_out = np.einsum('qrd,qdr->',env_out,np.conj(tens00_u))     # norm

        elif x_idx == (0,1):

            env_out = np.einsum('qQvV,qdl->QdlvV',env,np.conj(gam_u_dict[0,0]))
            env_out = np.einsum('QdlvV,QdL->vVlL',env_out,gam_d_dict[0,0])           # metric

            if idx_key == 'xo' or idx_key == 'oo':
                tens01_d = tf.dMult('MD',gam_d_dict[0,1],lam_d)    # qdx * lam(x)
                env_out = np.einsum('vVlL,VdL->vld', env_out,tens01_d)        # gradient
            if idx_key == 'oo':
                tens01_u = tf.dMult('MD',gam_u_dict[0,1],lam_u)    # qdx * lam(x)
                env_out = np.einsum('vld,vdl->',env_out,np.conj(tens01_u))     # norm

        else:  raise (IndexError)

    elif Ls == (2,1):
   
        if x_idx == (0,0): 

            env_out = np.einsum('qQvV,vdo->qQVdo',env,np.conj(gam_u_dict[1,0]))
            env_out = np.einsum('qQVdo,VdO->qQoO',env_out,gam_d_dict[1,0])          # metric

            if idx_key == 'xo' or idx_key == 'oo':
                tens00_d = tf.dMult('MD',gam_d_dict[0,0],lam_d)    # qdx * lam(x)
                env_out = np.einsum('qQoO,QdO->qod', env_out,tens00_d)       # gradient
            if idx_key == 'oo':
                tens00_u = tf.dMult('MD',gam_u_dict[0,0],lam_u)    # qdx * lam(x)
                env_out = np.einsum('qod,qdo->',env_out,np.conj(tens00_u))    # norm

        elif x_idx == (1,0):

            env_out = np.einsum('qQvV,qdi->QdivV',env,np.conj(gam_u_dict[0,0]))
            env_out = np.einsum('QdivV,QdI->vViI',env_out,gam_d_dict[0,0])          # metric

            if idx_key == 'xo' or idx_key == 'oo':
                tens10_d = tf.dMult('MD',gam_d_dict[1,0],lam_d)    # qdx * lam(x)
                env_out = np.einsum('vViI,VdI->vid', env_out,tens10_d)       # gradient
            if idx_key == 'oo':
                tens10_u = tf.dMult('MD',gam_u_dict[1,0],lam_u)    # qdx * lam(x)
                env_out = np.einsum('vid,vdi->',env_out,np.conj(tens10_u))    # norm

        else:  raise (IndexError)

    return env_out
