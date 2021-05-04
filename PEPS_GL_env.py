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

    xxx  note:  bonds/lambdas for legs sticking out are not included
    note:  bonds/lambdas for legs sticking out are  included
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
        # x_lam = [0,2]
        # ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=[0,3],op='sqrt')
        # ptens_u = PEPX_GL.apply_bond(ptens_u,peps_u_row.lambdas[j,3],3)
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=[0,3],op='sqrt')
        # ptens_d = PEPX_GL.apply_bond(ptens_d,peps_d_row.lambdas[j,3],3)

        # tens2 = np.einsum('liord,LIORd,xoOy->xlLiIyrR',ptens_u,ptens_d,boundary_mpo[j])
        tens2 = np.einsum('liord,xoOy->xliOyrd',ptens_u,boundary_mpo[j])
        tens2 = np.einsum('LIORd,xliOyrd->xlLiIyrR',ptens_d,tens2)
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


def get_next_boundary_I(peps_u_row,peps_d_row,boundary,XMAX=100):
    ''' get inside boundary (facing out)
        note:  peps_x_row is 1-D
    '''

    L2 = len(peps_u_row)
    boundary_mpo = boundary.copy()

    for j in range(L2):

        # x_lam = [1,3]
        # ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=[0,3],op='sqrt')
        # ptens_u = PEPX_GL.apply_bond(ptens_u,peps_u_row.lambdas[j,0],0)
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=[0,3],op='sqrt')
        # ptens_d = PEPX_GL.apply_bond(ptens_d,peps_d_row.lambdas[j,0],0)

        # print 'bI', np.linalg.norm(ptens_u), np.linalg.norm( PEPX_GL.get_site(peps_u_row,j,no_lam=[1]))

        # tens2 = np.einsum('xiIy,liord,LIORd->xlLoOyrR',boundary_mpo[j],peps_u_row[j],peps_d_row[j])
        tens2 = np.einsum('xiIy,liord->xlIoyrd',boundary_mpo[j],ptens_u)
        tens2 = np.einsum('xlIoyrd,LIORd->xlLoOyrR',tens2,ptens_d)
        boundary_mpo[j] = tf.reshape(tens2,'iii,i,i,iii')

    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,0)

    # print 'bi', [np.linalg.norm(m) for m in boundary_mpo]

    if np.sum(err) > 1.0:
        print 'bound I large compression error', XMAX
        print [np.linalg.norm(m) for m in boundary_mpo]
        print err
        # exit()

    return boundary_mpo, err


def get_next_boundary_L(peps_u_col,peps_d_col,boundary,XMAX=100):
    ''' get left boundary (facing right)
        note:  peps_x_col is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):

        # x_lam = [0,1]
        # ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)

        ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=[0,3],op='sqrt')
        # ptens_u = PEPX_GL.apply_bond(ptens_u,peps_u_col.lambdas[i,3],3)
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=[0,3],op='sqrt')
        # ptens_d = PEPX_GL.apply_bond(ptens_d,peps_d_col.lambdas[i,3],3)

        # tens2 = np.einsum('xlLy,liord,LIORd->xiIrRyoO',boundary_mpo[i],peps_u_col[i],peps_d_col[i])
        tens2 = np.einsum('xlLy,liord->xLioryd',boundary_mpo[i],ptens_u)
        tens2 = np.einsum('xLioryd,LIORd->xiIrRyoO',tens2,ptens_d)
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')
    
    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    boundary_mpo, err  = MPX.compress_reg(boundary_mpo,XMAX,0)

    return boundary_mpo, err


def get_next_boundary_R(peps_u_col,peps_d_col,boundary,XMAX=100):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    for i in range(L1):

        # x_lam = [2,3]
        # ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)

        ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=[0,3],op='sqrt')
        # ptens_u = PEPX_GL.apply_bond(ptens_u,peps_u_col.lambdas[i,0],0)
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=[0,3],op='sqrt')
        # ptens_d = PEPX_GL.apply_bond(ptens_d,peps_d_col.lambdas[i,0],0)


        # tens2 = np.einsum('liord,LIORd,xrRy->xiIlLyoO',peps_u_col[i],peps_d_col[i],boundary_mpo[i])
        tens2 = np.einsum('liord,xrRy->xlioRyd',ptens_u,boundary_mpo[i])
        tens2 = np.einsum('LIORd,xlioRyd->xiIlLyoO',ptens_d,tens2)
        boundary_mpo[i] = tf.reshape(tens2,'iii,i,i,iii')

    # err = np.nan
    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    boundary_mpo, err = MPX.compress_reg(boundary_mpo,XMAX,1)

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

def get_next_subboundary_O(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()

    for j in range(L2):

        # if j == L2-1:     x_lam = [0,2,3]
        # else:             x_lam = [0,2]

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # tens = np.einsum('liord,LIORd,xoOy->lLxiIrRy', peps_u_row[j],peps_d_row[j],e_mpo[j])
        tens = np.einsum('LIORd,xoOy->LxIoRyd',ptens_d,e_mpo[j])
        tens = np.einsum('liord,LxIoRyd->lLxiIrRy',ptens_u,tens)
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


def get_next_subboundary_I(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()
    
    for j in range(L2):

        # if j == 0:     x_lam = [0,1,3]
        # else:          x_lam = [1,3]

        # ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

        ptens_u = PEPX_GL.get_site(peps_u_row,j,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_row,j,op='sqrt')

        # tens = np.einsum('xiIy,liord,LIORd->xlLoOyrR', e_mpo[j],peps_u_row[j],peps_d_row[j])
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
    
    # err = np.nan
    e_mpo  = MPX.canonicalize(e_mpo,0)
    e_mpo, err = MPX.compress_reg(e_mpo,XMAX,1)

    return e_mpo, err


def get_next_subboundary_L(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100):

    L1 = len(peps_u_col)

    e_mpo = env_mpo.copy()
    
    for i in range(L1):

        # if i == L1-1:   x_lam = [0,1,2]
        # else:           x_lam = [0,1]   # looks right canonical

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)


        ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')

        # print 'bL', np.linalg.norm(ptens_u)
        # print 'bL', np.linalg.norm(PEPX_GL.get_site(peps_u_col,i,no_lam=[]))

        # print 'sub L', e_mpo[i].shape, peps_u_col[i].shape, peps_d_col[i].shape
        # tens = np.einsum('xlLy,liord,LIORd->xiIrRyoO', e_mpo[i],peps_u_col[i],peps_d_col[i])
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


def get_next_subboundary_R(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100):

    L1 = len(peps_u_col)
    e_mpo = env_mpo.copy()



    for i in range(L1):

        # if i == 0:      x_lam = [1,2,3]
        # else:           x_lam = [2,3]           # looks left canonical

        # ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
        # ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)


        ptens_u = PEPX_GL.get_site(peps_u_col,i,op='sqrt')
        ptens_d = PEPX_GL.get_site(peps_d_col,i,op='sqrt')


        # print 'br',i,L1,e_mpo[i].shape, peps_u_col[i].shape, peps_d_col[i].shape
        # tens = np.einsum('liord,LIORd,xrRy->iIxlLoOy', peps_u_col[i],peps_d_col[i],e_mpo[i])
        tens = np.einsum('LIORd,xrRy->xILryOd',ptens_d,e_mpo[i])
        tens = np.einsum('liord,xILryOd->xiIlLyoO',ptens_u,tens)
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
####  apply lambdas to boundary mpos    #####
#############################################


def apply_lam_to_boundary_mpo(b_mpo,bonds_u,bonds_d,op='sqrt_inv'):

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

    # ## lambda are not included on uncontracted bonds
    # b2_ = np.einsum('ludr,uU->lUdr',bound2[0],bonds_u[0])
    # b2_ = np.einsum('ludr,dD->luDr',b2_,bonds_d[0])
    # output = np.einsum('ludr,LudR->lLrR',bound1[0],b2_)
    # for m in range(1,len(bound1)):
    #     b2_ = np.einsum('ludr,uU->lUdr',bound2[m],bonds_u[m])
    #     b2_ = np.einsum('ludr,dD->luDr',b2_,bonds_d[m])
    #     output = np.einsum('lLrR,ruds,RudS->lLsS',output,bound1[m],b2_)

    # ## lambda are included on uncontracted bonds
    # b2_ = np.einsum('ludr,uU->lUdr',bound2[0],1./bonds_u[0])
    # b2_ = np.einsum('ludr,dD->luDr',b2_,1./bonds_d[0])
    # output = np.einsum('ludr,LudR->lLrR',bound1[0],b2_)
    # for m in range(1,len(bound1)):
    #     b2_ = np.einsum('ludr,uU->lUdr',bound2[m],1./bonds_u[m])
    #     b2_ = np.einsum('ludr,dD->luDr',b2_,1./bonds_d[m])
    #     output = np.einsum('lLrR,ruds,RudS->lLsS',output,bound1[m],b2_)

    # 
    # return np.einsum('llss->',output)

    ## sqrt lambdas are included on uncontracted bonds
    return ENV.contract_2_bounds(bound1,bound2)


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
        print 'in env_rho contract:  please use pepo or pep0'
        exit()
    
    if not get_err:   
        bound = boundaries[upto]

        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        return ovlp
    else:
        bounds, errs = boundaries
        bound = bounds[upto]
       
        ovlp = ovlp_from_bound(bound,to_squeeze=False)

        return ovlp, errs




#############################################################
####          embedded methods (on the fly env)          ####
#############################################################

def embed_sites_norm(sub_pepx,envs_list,side='L',XMAX=100,get_errs=False):
    ''' get norm of sub_pepx embedded in env '''

    # out =  embed_sites_ovlp( np.conj(PEPX.transposeUD(sub_pepx)), sub_pepx, envs_list, side, XMAX, get_errs)
    out =  embed_sites_ovlp( np.conj(sub_pepx), sub_pepx, envs_list, side, XMAX, get_errs)
    if get_errs:
        norm2, errs = out
        return np.sqrt(norm2), errs
    else:
        return np.sqrt(out)


def embed_sites_ovlp(sub_pepx_u_gl, sub_pepx_d_gl, envs_list, side='L',XMAX=100,get_errs=False):
    ''' get ovlp of sub_pepx_u, sub_pepx_d embedded in env '''


    # sub_pepx_u = PEPX_GL.get_pepx(sub_pepx_u_gl, outer_lam=False)
    # sub_pepx_d = PEPX_GL.get_pepx(sub_pepx_d_gl, outer_lam=False)

    sub_pepx_u = PEPX_GL.get_pepx(sub_pepx_u_gl, outer_lam='sqrt')
    sub_pepx_d = PEPX_GL.get_pepx(sub_pepx_d_gl, outer_lam='sqrt')

    # if sub_pepx_u_gl.shape == (1,2):
    #     temp_u1 = PEPX_GL.apply_lam_to_site(sub_pepx_u_gl[0,0],sub_pepx_u_gl.lambdas[0,0],no_lam=[3])
    #     temp_u1 = PEPX_GL.apply_bond(temp_u1,np.sqrt(sub_pepx_u_gl.lambdas[0,0,3]),3)
    #     temp_u2 = PEPX_GL.apply_lam_to_site(sub_pepx_u_gl[0,1],sub_pepx_u_gl.lambdas[0,1],no_lam=[0])
    #     temp_u2 = PEPX_GL.apply_bond(temp_u2,np.sqrt(sub_pepx_u_gl.lambdas[0,1,0]),0)

    #     print sub_pepx_u_gl.lambdas[0,0]
    #     print sub_pepx_u_gl.lambdas[0,1]
    #     print 'ovlp', np.linalg.norm(sub_pepx_u[0,0] - temp_u1)
    #     print 'ovlp', np.linalg.norm(sub_pepx_u[0,1] - temp_u2)

    # ovlp1 = ENV.embed_sites_ovlp(sub_pepx_u, sub_pepx_d, envs_list, side, XMAX, get_errs)

    # env = ENV.build_env(envs_list)
    # ovlp2 = ENV.embed_sites_ovlp_env(sub_pepx_u, sub_pepx_d, env)

    # print 'norms', ovlp1, ovlp2
    # return ovlp1

    return ENV.embed_sites_ovlp(sub_pepx_u, sub_pepx_d, envs_list, side, XMAX, get_errs)
 

##################################
# embedding:  keeping sites  #
##################################


def embed_sites_xo(sub_peps_u_gl, sub_peps_d_gl, envs, x_idx):
    ''' returns embedded system with site x_idx missing in bra
        (empty site in bra (u) , filled in ket (d))
    '''
    # sub_peps_u = PEPX.GL.get_pepx(sub_peps_u_gl, outer_lam=False)
    # sub_peps_d = PEPX.GL.get_pepx(sub_peps_d_gl, outer_lam=False)
    
    sub_peps_u = PEPX_GL.get_pepx(sub_peps_u_gl, outer_lam='sqrt')
    sub_peps_d = PEPX_GL.get_pepx(sub_peps_d_gl, outer_lam='sqrt')
    
    return ENV.embed_sites_xo(sub_peps_u,sub_peps_d,envs,x_idx)



def embed_sites_xx(sub_peps_u_gl, sub_peps_d_gl, envs, x_idx):
    ''' returns embedded system with site x_idx missing in both bra+ket
        update_conn denotes which bond was updated with trotter step
        (eg if 2x2 embedding, and only one bond was updated)
    '''

    # sub_peps_u = PEPX.GL.get_pepx(sub_peps_u_gl, outer_lam=False)
    # sub_peps_d = PEPX.GL.get_pepx(sub_peps_d_gl, outer_lam=False)

    sub_peps_u = PEPX_GL.get_pepx(sub_peps_u_gl, outer_lam='sqrt')
    sub_peps_d = PEPX_GL.get_pepx(sub_peps_d_gl, outer_lam='sqrt')

    return ENV.embed_sites_xx(sub_peps_u,sub_peps_d,envs,x_idx)


##################################
#   embedding:  keeping sites    #
#     with precomputed env       #
##################################


def build_env(envs,ensure_pos=False):
    # env MPOs already do NOT have lambdas on open bonds
    return ENV.build_env(envs,ensure_pos)


def embed_sites_xo_env(sub_peps_u_gl, sub_peps_d_gl, env_tens, x_idx):

    # sub_peps_u = PEPX.GL.get_pepx(sub_peps_u_gl, outer_lam=False)
    # sub_peps_d = PEPX.GL.get_pepx(sub_peps_d_gl, outer_lam=False)
    
    sub_peps_u = PEPX_GL.get_pepx(sub_peps_u_gl, outer_lam='sqrt')
    sub_peps_d = PEPX_GL.get_pepx(sub_peps_d_gl, outer_lam='sqrt')
    
    return ENV.embed_sites_xo_env(sub_peps_u,sub_peps_d,envs,x_idx)


def embed_sites_xx_env(sub_peps_u_gl, sub_peps_d_gl, env_tens, x_idx):

    # sub_peps_u = PEPX.GL.get_pepx(sub_peps_u_gl, outer_lam=False)
    # sub_peps_d = PEPX.GL.get_pepx(sub_peps_d_gl, outer_lam=False)
    
    sub_peps_u = PEPX_GL.get_pepx(sub_peps_u_gl, outer_lam='sqrt')
    sub_peps_d = PEPX_GL.get_pepx(sub_peps_d_gl, outer_lam='sqrt')
    
    return ENV.embed_sites_xx_env(sub_peps_u,sub_peps_d,envs,x_idx)


def embed_sites_norm_env(sub_pepx_gl,env_tens,x_idx=(0,0)):
    ''' get norm of sub_pepx embedded in env '''

    # sub_pepx = PEPX_GL.get_pepx(sub_pepx_gl, outer_lam=False)
    sub_pepx = PEPX_GL.get_pepx(sub_pepx_gl, outer_lam='sqrt')
    out =  ENV.embed_sites_ovlp_env( PEPX.conj_transpose(sub_pepx),sub_pepx,env_tens,x_idx)
    return np.sqrt(out)


def embed_sites_ovlp_env(sub_pepx_u_gl, sub_pepx_d_gl, env_tens, x_idx=(0,0)):
    ''' get ovlp of sub_pepx_u, sub_pepx_d embedded in env '''

    sub_peps_u = PEPX_GL.get_pepx(sub_peps_u_gl, outer_lam='sqrt')
    sub_peps_d = PEPX_GL.get_pepx(sub_peps_d_gl, outer_lam='sqrt')

    # sub_peps_u = PEPX.GL.get_pepx(sub_peps_u_gl, outer_lam=False)
    # sub_peps_d = PEPX.GL.get_pepx(sub_peps_d_gl, outer_lam=False)

    env_xo = ENV.embed_sites_xo_env(sub_pepx_u, sub_pepx_d, env_tens, x_idx)
    ovlp = np.einsum('liord,liord->',env_xo,sub_pepx_u[x_idx])

    return ovlp


