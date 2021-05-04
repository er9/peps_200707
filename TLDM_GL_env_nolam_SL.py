import numpy as np
import time
import tens_fcts as tf

import MPX
import PEPX
import PEPX_GL
import PEPS_GL_env_nolam_SL as ENV_GLS




def make_delta_phys_11(D_h,D_v,db):
    delta_virt = np.outer(np.eye(D_h),np.eye(D_v)).reshape(D_h,D_h,D_v,D_v)
    delta_virt = delta_virt.transpose(0,2,3,1)

    delta_phys = np.eye(db).reshape(db,db,1,1)    # diagonal from (1,0) to (0,1) --> R-I
    site = np.einsum('ABCD,abcd->AaBbCcDd',delta_virt,delta_phys)
    site = tf.reshape(site,'ii,ii,ii,ii')
    return site


def make_delta_phys_00(D_h,D_v,db):
    delta_virt = np.outer(np.eye(D_h),np.eye(D_v)).reshape(D_h,D_h,D_v,D_v)
    delta_virt = delta_virt.transpose(0,2,3,1)

    delta_phys = np.eye(db).reshape(1,1,db,db)    # diagonal from (1,0) to (0,1) --> I-R
    site = np.einsum('ABCD,abcd->AaBbCcDd',delta_virt,delta_phys)
    site = tf.reshape(site,'ii,ii,ii,ii')
    return site


def DL_to_SL_bound(peps_u,peps_d,orientation):

    L = len(peps_u)

    peps_new_u = []
    peps_new_d = []
    for i in range(L):
        DLu,DIu,DOu,DRu,dbP,dbA = peps_u[i].shape
        DLd,DId,DOd,DRd,dbP,dbA = peps_d[i].shape

        siteu = peps_u[i].transpose(0,5,1,2,4,3).reshape(DLu*dbA,DIu,DOu*dbP,DRu)
        sited = peps_d[i].transpose(0,1,5,2,3,4).reshape(DLd,DId*dbA,DOd,DRd*dbP)

        if orientation == 'col':
            peps_new_d += [make_delta_phys_00(DLu,DId,dbA)]   # actually no phys bond absorbed
            peps_new_d += [sited]

            peps_new_u += [siteu]
            peps_new_u += [make_delta_phys_11(DRd,DOu,dbP)] 

        elif orientation == 'row':
            peps_new_d += [sited]
            peps_new_d += [make_delta_phys_11(DRd,DOu,dbP)]   # actually no phys bond absorbed

            peps_new_u += [make_delta_phys_00(DLu,DId,dbA)] 
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
    return ENV_GLS.get_next_boundary_O(peps_u_row,peps_d_row,boundary,XMAX,scaleX,DS_fn=DL_to_SL_bound)

def get_next_boundary_I(peps_u_row,peps_d_row,boundary,XMAX=100,scaleX=1):
    ''' get inside boundary (facing out)
        note:  peps_x_row is 1-D
    '''
    return ENV_GLS.get_next_boundary_I(peps_u_row,peps_d_row,boundary,XMAX,scaleX,DS_fn=DL_to_SL_bound)


def get_next_boundary_L(peps_u_col,peps_d_col,boundary,XMAX=100,scaleX=1):
    ''' get left boundary (facing right)
        note:  peps_x_col is 1-D
    '''
    return ENV_GLS.get_next_boundary_L(peps_u_col,peps_d_col,boundary,XMAX,scaleX,DS_fn=DL_to_SL_bound)


def get_next_boundary_R(peps_u_col,peps_d_col,boundary,XMAX=100,scaleX=1):
    ''' get right boundary (facing left)
        note:  pepx_next is 1-D
    '''
    return ENV_GLS.get_next_boundary_R(peps_u_col,peps_d_col,boundary,XMAX,scaleX,DS_fn=DL_to_SL_bound)


#####################
#### full method ####
#####################

def get_boundaries(peps_u,peps_d,side,upto,init_bound=None,XMAX=100,scaleX=1,get_err=False):
    ''' get_boundaries for pep0 (no dangling physical bonds, e.g. traced out) 
    '''
    return ENV_GLS.get_boundaries(peps_u,peps_d,side,upto,init_bound,XMAX,scaleX,get_err,
                                  DS_fn=DL_to_SL_bound)



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

def get_next_subboundary_O(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100,c_tol=1.0e-8,scaleX=1):

    return ENV_GLS.get_next_subboundary_O(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX,c_tol,scaleX,
                                          DS_fn=DL_to_SL_bound)

def get_next_subboundary_I(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX=100,c_tol=1.0e-8,scaleX=1):

    return ENV_GLS.get_next_subboundary_I(env_mpo,boundL_tens,peps_u_row,peps_d_row,boundR_tens,XMAX,c_tol,scaleX,
                                          DS_fn=DL_to_SL_bound)

def get_next_subboundary_L(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100,c_tol=1.0e-8,scaleX=1):

    return ENV_GLS.get_next_subboundary_L(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX,c_tol,scaleX,
                                          DS_fn=DL_to_SL_bound)

def get_next_subboundary_R(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX=100,c_tol=1.0-8,scaleX=1):

    return ENV_GLS.get_next_subboundary_R(env_mpo,boundI_tens,peps_u_col,peps_d_col,boundO_tens,XMAX,c_tol,scaleX,
                                          DS_fn=DL_to_SL_bound)


#####################
#### full method ####
#####################


## need to fix to get correct env bounds
def get_subboundaries(bound1,bound2,peps_u_sub,peps_d_sub,side,upto,init_sb=None,XMAX=100,c_tol=1.0e-8,scaleX=1,
                      get_errs=False):
 
    return ENV_GLS.get_subboundaries(bound1,bound2,peps_u_sub,peps_d_sub,side,upto,init_sb,XMAX,c_tol,scaleX,get_errs,
                                     DS_fn=DL_to_SL_bound)  


#######################################################
# single layer boundaries --> double layer boundaries #
#######################################################

def SL_to_DL_bound(SL_bound, orientation):
    # SL bound -- len 2L  --> DL bound -- len L

    return ENV_GLS.SL_to_DL_bound(SL_bound, orientation)



#############################################
#####     ovlps from boundaries     #########
#############################################

def ovlp_from_bound(bound,to_squeeze=True):
    ''' obtain ovlp from bound which is from contracting entire peps by row/col
        should be mpo with physical bonds (1,1)
    '''
    return ENV_GLS.ovlp_from_bound(bound,to_squeeze)


def contract_2_bounds(bound1, bound2, bonds_u, bonds_d, orientation):
    ''' eg. bL -- bR, bI -- bO '''

    return ENV_GLS.contract_2_bounds(bound1,bound2,bonds_u,bonds_d,orientation)


def get_norm(tldm, side='I',XMAX=100,scaleX=1,get_err=False):

    norm2 = get_ovlp(np.conj(tldm), tldm, side, XMAX, scaleX, get_err=get_err)
    return np.sqrt(np.squeeze(norm2))


def get_ovlp(bra, ket, side='I',XMAX=100,scaleX=1,get_err=False):
    ''' contract virtual bonds of pep0 (no physical bonds) '''
   
    L1, L2 = ket.shape

    if   side in ['i','I',1]:   upto = L1
    elif side in ['l','L',0]:   upto = L2
    else:                       upto = 0

    boundaries = get_boundaries(bra,ket,side,upto,XMAX=XMAX,scaleX=scaleX,get_err=get_err)

    if not get_err:   
        bound = boundaries[-1]
        ovlp = ovlp_from_bound(bound,to_squeeze=False)
        return ovlp
    else:
        bounds, errs = boundaries
        bound = bounds[-1]
        ovlp = ovlp_from_bound(bound,to_squeeze=False)
        return ovlp, errs


def get_sub_ovlp(sub_pepx_u, sub_pepx_d, bounds, side=None, XMAX=100, scaleX=1, get_err=False):
    return ENV_GLS.get_sub_ovlp(sub_pepx_u, sub_pepx_d, bounds, side, XMAX, scaleX, get_err, sb_fn=get_subboundaries)
