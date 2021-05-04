import numpy as np
import time
import tens_fcts as tf

import MPX
import PEPX
import PEPS_env as ENV



def make_delta_phys_11(D_h,D_v,db):
    delta_virt = np.outer(np.eye(D_h),np.eye(D_v)).reshape(D_h,D_h,D_v,D_v)
    delta_virt = delta_virt.transpose(0,2,3,1)

    delta_phys = np.eye(db).reshape(db,db,1,1)    # diagonal from (1,0) to (0,1) --> R-I
    site = np.einsum('ABCD,abcd->AaBbCcDd',delta_virt,delta_phys)
    site = tf.reshape(site,'ii,ii,ii,ii')
    return site


def DL_to_SL_bound(peps_u,peps_d,orientation):

    L = len(peps_u)

    peps_new_u = []
    peps_new_d = []
    for i in range(L):
        DLu,DIu,DOu,DRu,db = peps_u[i].shape
        DLd,DId,DOd,DRd,db = peps_d[i].shape

        siteu = peps_u[i].transpose(0,1,2,4,3).reshape(DLu,DIu,DOu*db,DRu)
        sited = peps_d[i].transpose(0,1,2,3,4).reshape(DLd,DId,DOd,DRd*db)

        if orientation == 'col':
            peps_new_d += [make_delta_phys_11(DLu,DId,1)]   # actually no phys bond absorbed
            peps_new_d += [sited]

            peps_new_u += [siteu]
            peps_new_u += [make_delta_phys_11(DRd,DOu,db)] 

        elif orientation == 'row':
            peps_new_d += [sited]
            peps_new_d += [make_delta_phys_11(DRd,DOu,db)]   # actually no phys bond absorbed

            peps_new_u += [make_delta_phys_11(DLu,DId,1)] 
            peps_new_u += [siteu]

    return MPX.MPX(peps_new_u), MPX.MPX(peps_new_d)



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

def get_next_boundary_O(peps_u_row,peps_d_row,boundary,XMAX=100,scaleX=1):
    '''  get outside boundary (facing in)
         note:  peps_x_row is 2-D, but a 1 x L2 pepx_GL
    '''
    
    L2 = len(peps_u_row)
    boundary_mpo = boundary.copy()

    # # build the single layer rows
    # tens_list_u = []
    # tens_list_d = []
    # for j in range(L2):
    #     # ptens_u = peps_u_row[j]
    #     # ptens_d = peps_d_row[j]

    #     tens_list_u += [peps_u_row[j]]
    #     tens_list_d += [peps_d_row[j]]

    # row_u, row_d = DS_fn(tens_list_u,tens_list_d,'row')

    row_u, row_d = DL_to_SL_bound(peps_u_row,peps_d_row,'row')

    # contract the 2 rows (one row at a time)
    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('LIOR,xOy->LxIRy',row_d[j],boundary_mpo[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    # boundary_mpo, err1 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err1 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('LIOR,xOy->LxIRy',row_u[j],boundary_mpo[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    # boundary_mpo, err2 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err2 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    err = [err1[j]+err2[j] for j in range(len(err1))]

    if np.sum(err)/L2 > 1.0e-1:
        print 'norms', [np.linalg.norm(m) for m in boundary_mpo]
        print 'err1', err1
        print 'err2', err2
        raise RuntimeWarning('bound O large compression error'+str(XMAX))

    return boundary_mpo, err


def get_next_boundary_I(peps_u_row,peps_d_row,boundary,XMAX=100,scaleX=1):
    ''' get inside boundary (facing out)
        note:  peps_x_row is 1-D
    '''

    L2 = len(peps_u_row)
    boundary_mpo = boundary.copy()

    # # build the single layer rows
    # tens_list_u = []
    # tens_list_d = []
    # for j in range(L2):
    #     x_lam = [2,3]
    #     ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

    #     tens_list_u += [peps_u_row[j]]
    #     tens_list_d += [peps_d_row[j]]

    # row_u, row_d = DS_fn(tens_list_u,tens_list_d,'row')

    row_u, row_d = DL_to_SL_bound(peps_u_row,peps_d_row,'row')

    # contract the 2 rows (one row at a time)
    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('xIy,LIOR->xLOyR',boundary_mpo[j],row_u[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    # boundary_mpo, err1 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err1 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('xIy,LIOR->xLOyR',boundary_mpo[j],row_d[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    # boundary_mpo, err2 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err2 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    err = [err1[j]+err2[j] for j in range(len(err1))]

    if np.sum(err)/L2 > 1.0e-1:
        print 'norm',[np.linalg.norm(m) for m in boundary_mpo]
        print 'err1',err1
        print 'err2',err2
        raise RuntimeWarning('bound I large compression error'+str(XMAX))

    return boundary_mpo, err


def get_next_boundary_L(peps_u_col,peps_d_col,boundary,XMAX=100,scaleX=1):
    ''' get left boundary (facing right)
        note:  peps_x_col is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    # # build the single layer rows
    # tens_list_u = []
    # tens_list_d = []
    # for j in range(L1):
    #     x_lam = [1,3]
    #     ptens_u = PEPX_GL.get_site(peps_u_col,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_col,j,no_lam=x_lam)

    #     tens_list_u += [ptens_u]
    #     tens_list_d += [ptens_d]

    # col_u, col_d = DS_fn(tens_list_u,tens_list_d,'col')
    col_u, col_d = DL_to_SL_bound(peps_u_col,peps_d_col,'col')

    # contract the 2 rows (one row at a time)
    for j in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('xLy,LIOR->xIRyO',boundary_mpo[j],col_d[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    # boundary_mpo, err1 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err1 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    for j in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('xLy,LIOR->xIRyO',boundary_mpo[j],col_u[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,1)
    # boundary_mpo, err2 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err2 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    err = [err1[j]+err2[j] for j in range(len(err1))]

    if np.sum(err)/L1 > 1.0e-1:
        print 'norms', [np.linalg.norm(m) for m in boundary_mpo]
        print 'err1', err1
        print 'err2', err2
        raise RuntimeWarning('bound L large compression error'+str(XMAX))

    return boundary_mpo, err


def get_next_boundary_R(peps_u_col,peps_d_col,boundary,XMAX=100,scaleX=1):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''

    L1 = len(peps_u_col)
    boundary_mpo = boundary.copy()

    # # build the single layer rows
    # tens_list_u = []
    # tens_list_d = []
    # for j in range(L1):
    #     x_lam = [0,2]
    #     ptens_u = PEPX_GL.get_site(peps_u_col,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_col,j,no_lam=x_lam)

    #     tens_list_u += [ptens_u]
    #     tens_list_d += [ptens_d]

    # col_u, col_d = DS_fn(tens_list_u,tens_list_d,'col')
    col_u, col_d = DL_to_SL_bound(peps_u_col,peps_d_col,'col')

    # contract the 2 rows (one row at a time)
    for j in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('LIOR,xRy->xILyO',col_u[j],boundary_mpo[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    # boundary_mpo, err1 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err1 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    for j in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        tens2 = np.einsum('LIOR,xRy->xILyO',col_d[j],boundary_mpo[j])
        boundary_mpo[j] = tf.reshape(tens2,'ii,i,ii')

    boundary_mpo  = MPX.canonicalize(boundary_mpo,0)
    # boundary_mpo, err2 = MPX.compress_reg(boundary_mpo,XMAX,1)
    boundary_mpo, err2 = MPX.altD_compress(boundary_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    # print [m.shape for m in boundary_mpo]

    err = [err1[j]+err2[j] for j in range(len(err1))]

    if np.sum(err)/L1 > 1.0e-1:
        print 'norm',[np.linalg.norm(m) for m in boundary_mpo]
        print 'err1',err1
        print 'err2',err2
        raise RuntimeWarning('bound R large compression error'+str(XMAX))

    return boundary_mpo, err


#####################
#### full method ####
#####################

def get_boundaries(peps_u,peps_d,side,upto,init_bound=None,XMAX=100,scaleX=1,get_err=False,DS_fn=DL_to_SL_bound):
    ''' get_boundaries for pep0 (no dangling physical bonds, e.g. traced out) 
    '''

    # print 'peps-env-sl get boundaries', XMAX,scaleX

    L1, L2 = peps_u.shape

    if side in ['o','O',2]:

        if init_bound is None:        envs = [ MPX.ones([(1,)]*L2*2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]

        for i in range(L1-1,upto-1,-1):      # building envs from outside to inside
            boundary_mpo, err = get_next_boundary_O(peps_u[i,:],peps_d[i,:],envs[-1],XMAX,scaleX)
            envs.append( boundary_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]
            
    elif side in ['i','I',1]:

        if init_bound is None:        envs = [ MPX.ones([(1,)]*L2*2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
     
        for i in range(upto):              # building envs from inside to outside
            boundary_mpo, err = get_next_boundary_I(peps_u[i,:],peps_d[i,:],envs[-1],XMAX,scaleX)
            envs.append( boundary_mpo )
            errs.append( err )
    
    
    elif side in ['l','L',0]:

        if init_bound is None:        envs = [ MPX.ones([(1,)]*L1*2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(upto):              # building envs from left to right
            boundary_mpo, err = get_next_boundary_L(peps_u[:,j],peps_d[:,j],envs[-1],XMAX,scaleX)
            envs.append( boundary_mpo )
            errs.append( err )
    
       
    elif side in ['r','R',3]:

        if init_bound is None:        envs = [ MPX.ones([(1,)]*L1*2) ]     # initialize with empty boundary
        else:                         envs = [ init_bound ]
        errs = [ 0. ]
    
        for j in range(L2-1,upto-1,-1):      # building envs from right to left
            boundary_mpo, err = get_next_boundary_R(peps_u[:,j],peps_d[:,j],envs[-1],XMAX,scaleX)
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

def get_next_subboundary_O(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100,scaleX=1):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()

    # # make single layer rows
    # tens_list_u = []
    # tens_list_d = []
    # for j in range(L2):

    #     if j == 0:     x_lam = [1] 
    #     else:          x_lam = [0,1]

    #     ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

    #     tens_list_u += [ptens_u]
    #     tens_list_d += [ptens_d]

    # row_u, row_d = DS_fn( tens_list_u,tens_list_d,'row')
    row_u, row_d = DL_to_SL_bound(peps_u_row,peps_d_row,'row')


    # contract envs with boundary 1, boundary 2
    # contract 2 subrows
    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[j] = np.einsum('LIOR,xOy->LxIRy',row_d[j],e_mpo[j])

    e_mpo[0]  = np.einsum('wLx,LxIRy->wIRy',boundL_tens[1],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('LxIRy,wRy->LxIw',e_mpo[-1],boundR_tens[1])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for j in range(1,2*L2-1):
        # e_mpo[j] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[j] = tf.reshape(e_mpo[j],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,1)
    # e_mpo, err1 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err1 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[j] = np.einsum('LIOR,xOy->LxIRy',row_u[j],e_mpo[j])

    e_mpo[0]  = np.einsum('wLx,LxIRy->wIRy',boundL_tens[0],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('LxIRy,wRy->LxIw',e_mpo[-1],boundR_tens[0])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for j in range(1,2*L2-1):
        # e_mpo[j] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[j] = tf.reshape(e_mpo[j],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,1)
    # e_mpo, err2 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err2 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    err = [err1[j] + err2[j] for j in range(len(err1))]

    if np.sum(err)/L2 > 1.0e-1:
        print 'norm',[np.linalg.norm(m) for m in e_mpo]
        print 'err1',err1
        print 'err2',err2
        raise RuntimeWarning('subbound O large compression error'+str(XMAX))

    return e_mpo, err


def get_next_subboundary_I(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100,scaleX=1):

    L2 = len(peps_u_row)
    e_mpo = env_mpo.copy()
    
    # # make single layer rows
    # tens_list_u = []
    # tens_list_d = []
    # for j in range(L2):

    #     if j == L2-1:     x_lam = [2]
    #     else:             x_lam = [2,3]

    #     ptens_u = PEPX_GL.get_site(peps_u_row,j,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_row,j,no_lam=x_lam)

    #     tens_list_u += [ptens_u]
    #     tens_list_d += [ptens_d]

    # row_u, row_d = DS_fn( tens_list_u,tens_list_d,'row')
    row_u, row_d = DL_to_SL_bound(peps_u_row,peps_d_row,'row')

    # contract envs with boundary 1, boundary 2
    # contract 2 subrows
    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[j] = np.einsum('xIy,LIOR->xLOyR',e_mpo[j],row_u[j])

    e_mpo[0]  = np.einsum('xLw,xLOyR->wOyR',boundL_tens[0],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('xLOyR,yRw->xLOw',e_mpo[-1],boundR_tens[0])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for j in range(1,2*L2-1):
        # e_mpo[j] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[j] = tf.reshape(e_mpo[j],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,0)
    # e_mpo, err1 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err1 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    for j in range(2*L2):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[j] = np.einsum('xIy,LIOR->xLOyR',e_mpo[j],row_d[j])

    e_mpo[0]  = np.einsum('xLw,xLOyR->wOyR',boundL_tens[1],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('xLOyR,yRw->xLOw',e_mpo[-1],boundR_tens[1])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for j in range(1,2*L2-1):
        # e_mpo[j] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[j] = tf.reshape(e_mpo[j],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,0)
    # e_mpo, err2 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err2 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    err = [err1[j] + err2[j] for j in range(len(err1))]

    if np.sum(err)/L2 > 1.0e-1:
        print 'norm',[np.linalg.norm(m) for m in e_mpo]
        print 'err1',err1
        print 'err2'.err2
        raise RuntimeWarning('subbound I large compression error'+str(XMAX))

    return e_mpo, err


def get_next_subboundary_L(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100,scaleX=1):

    L1 = len(peps_u_col)
    e_mpo = env_mpo.copy()

    # col_u, col_d = DS_fn( tens_list_u, tens_list_d, 'col')
    col_u, col_d = DL_to_SL_bound(peps_u_col,peps_d_col,'col')


    # contract envs with boundary 1, boundary 2
    # contract 2 subrows
    for j in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[j] = np.einsum('xLy,LIOR->xIRyO',e_mpo[j],col_d[j])

    e_mpo[0]  = np.einsum('xIw,xIRyO->wRyO',boundI_tens[0],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('xIRyO,yOw->xIRw',e_mpo[-1],boundO_tens[0])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for i in range(1,2*L1-1):
        # e_mpo[i] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[i] = tf.reshape(e_mpo[i],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,0)
    # e_mpo, err1 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err1 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    for i in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[i] = np.einsum('xLy,LIOR->xIRyO',e_mpo[i],col_u[i])

    # print [m.shape for m in e_mpo]

    e_mpo[0]  = np.einsum('xIw,xIRyO->wRyO',boundI_tens[1],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('xIRyO,yOw->xIRw',e_mpo[-1],boundO_tens[1])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for i in range(1,2*L1-1):
        # e_mpo[i] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[i] = tf.reshape(e_mpo[i],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,0)
    # e_mpo, err2 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err2 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),1,regularize=True)

    err = [err1[i] + err2[i] for i in range(len(err1))]

    if np.sum(err)/L1 > 1.0e-1:
        print 'norm',[np.linalg.norm(m) for m in e_mpo]
        print 'err1',err1
        print 'err2',err2
        raise RuntimeWarning('subbound L large compression error'+str(XMAX))

    return e_mpo, err


def get_next_subboundary_R(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100,scaleX=1):

    L1 = len(peps_u_col)
    e_mpo = env_mpo.copy()


    # # make single layer rows
    # tens_list_u = []
    # tens_list_d = []
    # for i in range(L1):

    #     if i == L1-1:      x_lam = [0]
    #     else:              x_lam = [0,2]

    #     ptens_u = PEPX_GL.get_site(peps_u_col,i,no_lam=x_lam)
    #     ptens_d = PEPX_GL.get_site(peps_d_col,i,no_lam=x_lam)

    #     tens_list_u += [ptens_u]
    #     tens_list_d += [ptens_d]

    # col_u, col_d = DS_fn( tens_list_u, tens_list_d, 'col')
    col_u, col_d = DL_to_SL_bound(peps_u_col,peps_d_col,'col')


    # contract envs with boundary 1, boundary 2
    # contract 2 subrows
    for i in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[i] = np.einsum('LIOR,xRy->IxLOy',col_u[i],e_mpo[i])

    e_mpo[0]  = np.einsum('wIx,IxLOy->wLOy',boundI_tens[1],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('IxLOy,wOy->IxLw',e_mpo[-1],boundO_tens[1])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for i in range(1,2*L1-1):
        # e_mpo[i] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[i] = tf.reshape(e_mpo[i],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,1)
    # e_mpo, err1 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err1 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    for i in range(2*L1):       # assume sum(shape(bra)) < sum(shape(ket))
        e_mpo[i] = np.einsum('LIOR,xRy->IxLOy',col_d[i],e_mpo[i])

    e_mpo[0]  = np.einsum('wIx,IxLOy->wLOy',boundI_tens[0],e_mpo[0])
    e_mpo[0]  = tf.reshape(e_mpo[0],'i,i,ii')
    e_mpo[-1] = np.einsum('...LOy,wOy->...Lw',e_mpo[-1],boundO_tens[0])
    e_mpo[-1] = tf.reshape(e_mpo[-1],'ii,i,i')
    for i in range(1,2*L1-1):
        # e_mpo[i] = tf.reshape(tens2,'ii,i,ii')
        e_mpo[i] = tf.reshape(e_mpo[i],'ii,i,ii')

    e_mpo  = MPX.canonicalize(e_mpo,1)
    # e_mpo, err2 = MPX.compress_reg(e_mpo,XMAX,0)
    e_mpo, err2 = MPX.altD_compress(e_mpo,(scaleX*XMAX,XMAX),0,regularize=True)

    err = [err1[i] + err2[i] for i in range(len(err1))]

    if np.sum(err)/L1 > 1.0e-1:
        print 'norm',[np.linalg.norm(m) for m in e_mpo]
        print 'err1',err1
        print 'err2',err2
        raise RuntimeWarning('subbound R large compression error'+str(XMAX))

    return e_mpo, err



#####################
#### full method ####
#####################


## need to fix to get correct env bounds
def get_subboundaries(bound1,bound2,peps_u_sub,peps_d_sub,side,upto,init_sb=None,XMAX=100,scaleX=1,get_errs=False):
    
    L1, L2 = peps_u_sub.shape  

    if side in ['o','O',2]:    # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,)]*L2*2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(L1-1,upto-1,-1):   # get envs up to row upto
            ii = 2*i
            e_mpo, err = get_next_subboundary_O(envs[-1],bound1[ii:ii+2],peps_u_sub[i,:],peps_d_sub[i,:],
                                                bound2[ii:ii+2],XMAX,scaleX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]

    elif side in ['i','I',1]:   # bound1, bound2 are vertical mpos (left and right side)
        if init_sb is None:      envs = [ MPX.ones([(1,)]*L2*2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for i in range(upto):   # get envs up to row upto
            ii = 2*i
            e_mpo, err = get_next_subboundary_I(envs[-1],bound1[ii:ii+2],peps_u_sub[i,:],peps_d_sub[i,:],
                                                bound2[ii:ii+2],XMAX,scaleX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['l','L',0]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,)]*L1*2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(upto):   # get envs up to col upto
            jj = 2*j
            e_mpo, err = get_next_subboundary_L(envs[-1],bound1[jj:jj+2],peps_u_sub[:,j],peps_d_sub[:,j],
                                                bound2[jj:jj+2],XMAX,scaleX)
            envs.append( e_mpo )
            errs.append( err )

    elif side in ['r','R',3]:   # bound1, bound2 are horizontal mpos (top and bottom)
        if init_sb is None:      envs = [ MPX.ones([(1,)]*L1*2) ]
        else:                    envs = [ init_sb ]
        errs = [ 0. ]
        for j in range(L2-1,upto-1,-1):   # get envs up to col upto
            jj = 2*j

            e_mpo, err = get_next_subboundary_R(envs[-1],bound1[jj:jj+2],peps_u_sub[:,j],peps_d_sub[:,j],
                                                bound2[jj:jj+2],XMAX,scaleX)
            envs.append( e_mpo )
            errs.append( err )
        envs = envs[::-1]
        errs = errs[::-1]


    if get_errs:   return envs, errs
    else:          return envs


#######################################################
# single layer boundaries --> double layer boundaries #
#######################################################

def SL_to_DL_bound(SL_bound,orientation):
    # SL bound -- len 2L  --> DL bound -- len L

    L = len(SL_bound)/2

    tens_list = []
    if orientation == 'col':
        for x in range(L):
            tens_list.append( np.tensordot( SL_bound[2*x], SL_bound[2*x+1], axes=(-1,0)) )
    elif orientation == 'row':
        for x in range(L):
            temp = np.tensordot( SL_bound[2*x], SL_bound[2*x+1], axes=(-1,0))
            tens_list.append( temp.transpose(0,2,1,3) )

    return MPX.MPX( tens_list )



#############################################
#####     ovlps from boundaries     #########
#############################################

def ovlp_from_bound(bound,to_squeeze=True):
    ''' obtain ovlp from bound which is from contracting entire peps by row/col
        should be mpo with physical bonds (1,1)
    '''

    ovlp = np.einsum('lds,sdr->lr',bound[0],bound[1])
    for i in range(2,len(bound),2):
        ovlp = np.einsum('lR,Rds->lds',ovlp,bound[i])
        ovlp = np.einsum('lds,sdr->lr',ovlp,bound[i+1])

    if to_squeeze:
        return np.squeeze(ovlp)
    else:
        return ovlp


def contract_2_bounds(bound1, bound2, bonds_u, bonds_d, orientation):
    ''' eg. bL -- bR, bI -- bO '''

    ## lambda are not included on uncontracted bonds
    b2 = []

    if orientation == 'col':
        for i in range(0,len(bound1),2):
            b2.append( np.einsum('lur,uU->lUr',bound2[i]  ,np.diag(bonds_u[0])) )
            b2.append( np.einsum('ldr,dD->lDr',bound2[i+1],np.diag(bonds_d[0])) )
    elif orientation == 'row':
        for i in range(0,len(bound1),2):
            b2.append( np.einsum('lur,uU->lUr',bound2[i]  ,np.diag(bonds_d[0])) )
            b2.append( np.einsum('ldr,dD->lDr',bound2[i+1],np.diag(bonds_u[0])) )


    b2 = MPX.MPX(b2)

    return MPX.mps_dot(bound1,b2)


def get_ovlp(pepx_u, pepx_d, side='I',XMAX=100,scaleX=1,get_err=False):
    ''' contract virtual bonds of pep0 (no physical bonds) '''
   
    L1, L2 = pepx_u.shape

    if   side in ['i','I',1]:   upto = L1
    elif side in ['l','L',0]:   upto = L2
    else:                       upto = 0

    if np.all( [len(dp) == 1 for dp in pepx_u.phys_bonds.flat] ):
        boundaries = get_boundaries(pepx_u,pepx_d,side,upto,XMAX=XMAX,scaleX=scaleX,get_err=get_err)
    elif np.all( [len(dp) == 2 for dp in pepx_u.phys_bonds.flat] ):
        # flatten bonds 
        pepx_uf = PEPX.flatten(PEPX.transposeUD(pepx_u))
        pepx_df = PEPX.flatten(PEPX.transposeUD(pepx_d))
        boundaries = get_boundaries(pepx_uf,pepx_df,side,upto,XMAX=XMAX,scaleX=scaleX,get_err=get_err)
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


def get_norm(peps, side='I',XMAX=100, scaleX=1, get_err=False):

    norm2 = get_ovlp(np.conj(peps), peps, side, XMAX, scaleX, get_err=get_err)
    return np.sqrt(np.squeeze(norm2))


def get_sub_ovlp(sub_pepx_u, sub_pepx_d, bounds, side=None, XMAX=100, scaleX=1, get_err=False):

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
            bounds = get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bI,XMAX=XMAX,scaleX=scaleX,
                                       get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bO)
        elif side in ['o','O',2]:
            bounds = get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bO,XMAX=XMAX,scaleX=scaleX,
                                       get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bI)
        elif side in ['l','L',0]:
            bounds = get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bL,XMAX=XMAX,scaleX=scaleX,
                                      get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bL)
        elif side in ['r','R',3]:
            bounds = get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bR,XMAX=XMAX,scaleX=scaleX,
                                      get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bR)

        return ovlp

    else:
        if side   in ['i','I',1]:
            bounds,errs= get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bI,XMAX=XMAX,scaleX=scaleX,
                                           get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bO)
        elif side in ['o','O',2]:
            bounds,errs= get_subboundaries(bL,bR,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bO,XMAX=XMAX,scaleX=scaleX,
                                           get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bI)
        elif side in ['l','L',0]:
            bounds,errs= get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bL,XMAX=XMAX,scaleX=scaleX,
                                           get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bL)
        elif side in ['r','R',3]:
            bounds,errs= get_subboundaries(bI,bO,sub_pepx_u,sub_pepx_d,side,upto,init_sb=bR,XMAX=XMAX,scaleX=scaleX,
                                           get_errs=get_err)
            ovlp  = contract_2_bounds(bounds[upto], bR)

        return ovlp, errs
