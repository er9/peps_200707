''' two-layer thermal state code:  represent thermal state as entangled layer of 2 peps (+ ancilla leg)
 '''

import numpy as np
import copy
import time
import tens_fcts as tf

import MPX
import PEPX
import PEPX_GL
import PEPS_env as ENV
import PEPS_GL_env_nolam as ENV_GL

import Operator_2D as Op
import PEPS_env as penv
import PEP0_env as penv0

import TLDM_GL_env_nolam as ENV_DM
import TLDM_GL_env_nolam_SL as ENV_DMS


class TLDM_GL(PEPX_GL.PEPX_GL):
    """
    Lightweight PEPS/PEPO class in Gamma/Lambda decomposition

    2D array storing Gamma's
    2D array storing Lambda's
    2D array storing phys_bonds (same as before) (is it really necessary..?)

    """

    def __add__(tldm1, tldm2):
        return add(tldm1, tldm2)
    
    def __sub__(tldm1, tldm2):
        return add(tldm1, mul(-1,tldm2))

    def __mul__(tldm, alpha):
        return mul(alpha, tldm)

    def __neg__(tldm):
        return tldm.mul(-1)
    
    def dot(tldm1, tldm2):
        return dot(tldm2,tldm1)

    def norm(tldm):
        return norm(tldm)

    def transposeUD(tldm):
        return transposeUD(tldm)

    def copy(tldm):
        return copy(tldm)



###############################
###### create TFD states ######
###############################


def phys_bond_add_dim(tup_array):
    ''' (x,) --> (x,x) for all idx in array '''

    L1,L2 = len(tup_array),len(tup_array[0])

    new_array = np.empty(L1,L2,dtype=tuple)
    for idx in np.ndindex(L1,L2):
        new_array[idx] = tup_array[idx]*2

    return new_array


def eye(dp):    # maximally entangled
    ''' dp is 2D array of physical bond dimensions (d,) '''
    
    tp1 = PEPX_GL.eye(dp)
    return TLDM_GL(tp1,tp1.lambdas,tp1.phys_bonds)

    
def pure_product_state(dp,occ):   
    ''' dp, occ = 2D array of tuples (d,) '''

    tp1 = PEPX_GL.product_peps(dp,occ)

    return pure_state(tp1)


def random_pure_product_state(dp):
    ''' dp, occ = 2D array of tuples (d,) '''
    
    rand_mps = PEPX_GL.random_product_state(dp)
    
    return pure_state(rand_mps)


def pure_state(peps1):
    ''' take outer product of peps and its c.c. (into TLDM) '''

    # add ancilliary legs
    anc1 = np.empty(peps1.shape,dtype=np.object)

    for idx in np.ndindex(peps1.shape):
   
        sh1 = peps1[idx].shape
        tens1 = peps1[idx].reshape( sh1 + (1,) )
    
        anc1[idx] = tens1

    return TLDM_GL(anc1,peps1.lambdas)
    

def random_mixed_product_state(dp):
    ''' do = 2D array of tuples (d,) '''

    dp_ = phys_bond_add_dim(dp)

    rand_pepo = PEPX_GL.random_product_state(dp_)
    pepo_norm = PEPX_GL.norm(rand_pepo)
    
    tldm = TDLM_GL( PEPX_GL.mul(1./pepo_norm), rand_pepo )

    return tldm
    

#############################
####### other fcts? #########
#############################

def copy(tldm):
    # deeper copy than mpx.copy, as it copies the ndarrays in mpx

    new_tldm = np.empty(tldm.shape,dtype=np.object)
    new_lams = np.empty(tldm.lambdas.shape,dtype=np.object)

    for idx, tens in np.ndenumerate(tldm):
        new_tldm[idx] = tens.copy()
        new_lams[idx] = [m.copy() for m in tldm.lambdas[idx]]

    dps = tldm.phys_bonds.copy()

    return TLDM_GL(new_tldm,new_lams,dps)


def flatten(tldm):
    """   Converts PEPO object into PEPS    """

    peps = np.empty(tldm.shape,dtype=object)
    for ind in np.ndindex(tldm.shape):
        ptens = tldm[ind]
        if ptens.ndim == 5:  peps[ind] = ptens
        else:                peps[ind] = tf.reshape(ptens,'i,i,i,i,...',group_ellipsis=True)
    return TLDM_GL(peps,tldm.lambdas)


def unflatten(peps,dbs):
    '''  converts PEPS object into PEPO '''

    tldm = np.empty(peps.shape,dtype=object)
    for ind in np.ndindex(tldm.shape):
        ptens = peps[ind]
        tldm[ind] = ptens.reshape( ptens.shape[:4]+dbs[ind] )
    return TLDM_GL(tldm,peps.lambdas,dbs)

##############################
#### arithemtic functions ####
##############################

def cc_dot(pepo,tldm):
    ''' dot of pepo and c.c. to TLDM '''

    new_tldm = PEPX.dot(pepo,tldm)    

    return new_tldm


def vdot(tldm1, tldm2, side='I',XMAX=100,contract_SL=False,scaleX=1):
    if contract_SL:
        ovlp =  ENV_DMS.get_ovlp(np.conj(tldm1),tldm2,side,XMAX,scaleX)
    else:
        ovlp =  ENV_DM.get_ovlp(np.conj(tldm1),tldm2,side,XMAX)
    return np.squeeze(ovlp)


def norm(tldm,side='I',XMAX=100,contract_SL=False,scaleX=1):
    if contract_SL:
        return ENV_DMS.get_norm(tldm,side,XMAX=XMAX,scaleX=scaleX)
    else:
        return ENV_DM .get_norm(tldm,side,XMAX=XMAX)

def trace_norm(tldm,side='I',XMAX=100,contract_SL=False,scaleX=1):
    if contract_SL:
        return ENV_DMS.get_norm(tldm,side,XMAX=XMAX,scaleX=scaleX)
    else:
        return ENV_DM .get_norm(tldm,side,XMAX=XMAX)


def mul(alpha, tldm):
    """
    scales mpx by constant factor alpha (can be complex)
    """

    return PEPX_GL.mul(alpha,tldm)


def add(pepx1, pepx2, obc=(True,True,True,True)):
    """
    Direct sum of MPX's of the same shape
    obc:  if True, reshape MPX s.t. end dims are 1
    """

    ## will increase ancilla bond dimenson ##

    raise(NotImplementedError)


def meas_obs(tldm,op,ind0=(0,0),op_conn=None,side='I',XMAX=100,envs=None,return_norm=False,
             contract_SL=False,scaleX=1):
    # operator acts on sites ind0 : ind0+len(mpo)


    if isinstance(op,Op.TrotterOp):    # measure sum of small operators 
        return meas_obs_trotterH(tldm,op,XMAX,bounds=envs,return_norm=return_norm,contract_SL=contract_SL,
                                 scaleX=scaleX)

    elif isinstance(op,str):           # measure magnetization 
        return meas_obs_mag(tldm,pauli_op=op,bounds=envs,XMAX=XMAX,return_norm=return_norm,
                            contract_SL=contract_SL,scaleX=scaleX)

    elif isinstance(op,tuple):
        try:     ind0 = op[2]
        except:  ind0 = None    # defaults to L1/2, L2/2

        return meas_obs_corr(tldm,pauli_op=op[:2],ref_ind=ind0,bounds=envs,XMAX=XMAX,return_norm=return_norm,
                             contract_SL=contract_SL,scaleX=scaleX)

    elif isinstance(op,list):
        ops = op[0]
        idxs = op[1]
        return meas_obs_prod_op(tldm,ops,idxs,bounds=envs,XMAX=XMAX,return_norm=return_norm,
                                contract_SL=contract_SL,scaleX=scaleX)

    elif callable(op):                 # meaure function op
        if return_norm:
            return op(tldm), norm(tldm)
        else:
            return op(tldm)
    
    else:                              # measure PEPX or MPX
        tldm_p = PEPX_GL.PEPX_GL(tldm)

        # return PEPX_GL.meas_obs(tldm_p,op,ind0,op_conn,'peps',side,XMAX,envs_list,return_norm)

        if isinstance(op,PEPX.PEPX):
            return PEPX_GL.meas_obs(tldm_p,op,ind0,op_conn,'peps',side,XMAX,envs,return_norm,
                                    contract_SL=contract_SL,scaleX=scaleX)
            ### current implementation of embed ovlp in ENV. is not efficient

        elif isinstance(op,MPX.MPX):
            gams, lams, axT  = PEPX_GL.get_sites(tldm,ind0,op_conn)   
            gams, lams, errs = PEPX_GL.mpo_update(gams,lams,op,DMAX=-1,num_io=2,normalize=False)
    
            op_tldm = PEPX_GL.set_sites(tldm,ind0,op_conn,gams,lams,axT)

        if envs is not None:
            exp_val = ENV_DM.embed_sites_ovlp(np.conj(tldm),op_tldm,envs)
        else:
            if contract_SL:
                exp_val = ENV_DMS.get_ovlp(np.conj(tldm),op_tldm,side=side,XMAX=XMAX,scaleX=scaleX)
                exp_val = np.squeeze(exp_val)
            else:
                exp_val = ENV_DM.get_ovlp(np.conj(tldm),op_tldm,side=side,XMAX=XMAX)
                exp_val = np.squeeze(exp_val)

        if return_norm:
            norm_val = norm(tldm,contract_SL=contract_SL,scaleX=scaleX)
            exp_val = exp_val/(norm_val**2)
            return exp_val, norm_val
        else:
            return exp_val


### alternative method:  converts to pepx ###
# def meas_obs(tldm,op,ind0=(0,0),op_conn=None,envs_list=None,side='I',DMAX=10,XMAX=100,return_norm=False):
#     # operator acts on sites ind0 : ind0+len(mpo)
# 
# 
#     if isinstance(op,Op.TrotterOp):
#         return meas_obs_trotterH(tldm,op,DMAX,XMAX,return_norm=return_norm)
# 
#     else:
#         op_pepx = PEPX_GL.get_pepx(tldm)
#         pepx = PEPX_GL.get_pepx(tldm)
#     
#         if isinstance(op,PEPX.PEPX):
#             L1,L2 = op.shape
#             ix,iy = ind0
#     
#             op_pepx[ix:ix+L1,iy:iy+L2] = PEPX.dot(op,pepx[ix:ix+L1,iy:iy+L2])
#     
#         elif isinstance(op,MPX.MPX):
#             xs,ys = PEPX.get_conn_inds(op_conn,ind0)
# 
#             r1, r2 = min(xs), max(xs)
#             c1, c2 = min(ys), max(ys)
# 
#             pepx_list, axTs = PEPX.connect_pepx_list(pepx[xs,ys],op_conn)
#             app_list,  errs = PEPX.mpo_update(pepx_list,None,op,DMAX=-1)
# 
#             op_pepx[xs,ys] = PEPX.transpose_pepx_list(app_list, axTs)
# 
#         op_peps = PEPX.flatten(op_pepx)
#         peps = PEPX.flatten(pepx)
#     
#         if envs_list is not None:
#             exp_val = ENV.embed_sites_ovlp(np.conj(peps),op_peps,envs_list)
#         else:
#             exp_val = ENV.get_ovlp(np.conj(peps),op_peps,side=side,XMAX=XMAX)
#     
#         return exp_val

## get bounds as needed
def meas_get_bounds(tldm,bounds,op_shape,XMAX,contract_SL,scaleX=1):

    L1, L2 = tldm.shape
    NR, NC = op_shape

    # set which contraction scheme is used
    if contract_SL:
        return meas_get_bounds_SL(tldm,bounds,op_shape,XMAX,contract_SL,scaleX)

    # else, DL alg
    get_bounds = ENV_DM.get_boundaries
    get_subbounds = ENV_DM.get_subboundaries

    # calculate envs and sweep through
    if bounds is None:
        envIs = get_bounds( np.conj(tldm), tldm, 'I', L1, XMAX=XMAX)
        envOs = get_bounds( np.conj(tldm), tldm, 'O', 0 , XMAX=XMAX)

        senvRs = []
        senvLs = []
        if NR > 0:
            for i in range(L1):
                NR_ = min(NR,L1-i)
                senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                            tldm[i:i+NR_,:],'R',0 ,XMAX=XMAX))
                senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                            tldm[i:i+NR_,:],'L',L2,XMAX=XMAX))
    else:
        senvLs, envIs, envOs, senvRs = bounds

        if NR > 0:
            if senvLs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                                tldm[i:i+NR_,:],'R',0 ,XMAX=XMAX))

            if senvRs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                                tldm[i:i+NR_,:],'L',L2,XMAX=XMAX))

    # put into DL form if necessary
    return senvLs, envIs, envOs, senvRs


def meas_get_bounds_SL(tldm,bounds,op_shape,XMAX,contract_SL,scaleX):

    L1, L2 = tldm.shape
    NR, NC = op_shape

    # set which contraction scheme is used
    get_bounds = ENV_DMS.get_boundaries
    get_subbounds = ENV_DMS.get_subboundaries

    # calculate envs and sweep through
    if bounds is None:
        envIs = get_bounds( np.conj(tldm), tldm, 'I', L1, XMAX=XMAX,scaleX=scaleX)
        envOs = get_bounds( np.conj(tldm), tldm, 'O', 0 , XMAX=XMAX,scaleX=scaleX)

        senvRs = []
        senvLs = []
        if NR > 0:
            for i in range(L1):
                NR_ = min(NR,L1-i)
                senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                            tldm[i:i+NR_,:],'R',0 ,XMAX=XMAX,scaleX=scaleX))
                senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                            tldm[i:i+NR_,:],'L',L2,XMAX=XMAX,scaleX=scaleX))
    else:
        senvLs, envIs, envOs, senvRs = bounds

        if NR > 0:
            if senvLs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                                tldm[i:i+NR_,:],'R',0 ,XMAX=XMAX,scaleX=scaleX))

            if senvRs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(tldm[i:i+NR_,:]),
                                                tldm[i:iNR_,:],'L',L2,XMAX=XMAX,scaleX=scaleX))

    # put into DL form 
    tempIs, tempOs = [], []
    for i in range(len(envIs)):
        tempIs.append(ENV_DMS.SL_to_DL_bound(envIs[i],'row'))
        tempOs.append(ENV_DMS.SL_to_DL_bound(envOs[i],'row'))

    tempLs, tempRs = [], []
    for j1 in range(len(senvLs)):
        subLs, subRs = [], []
        for j2 in range(len(senvLs[j1])):
            subLs.append(ENV_DMS.SL_to_DL_bound(senvLs[j1][j2],'col'))
            subRs.append(ENV_DMS.SL_to_DL_bound(senvRs[j1][j2],'col'))
        tempLs.append([bm for bm in subLs])
        tempRs.append([bm for bm in subRs])

    return tempLs, tempIs, tempOs, tempRs


def meas_obs_trotterH(tldm,trotterH,XMAX=100,bounds=None,return_norm=False,contract_SL=False,scaleX=1):

    L1, L2 = trotterH.Ls
    NR, NC = trotterH.it_sh

    senvLs, envIs, envOs, senvRs = meas_get_bounds(tldm,bounds,(NR,NC),XMAX,contract_SL,scaleX)


    # make measurements
    obs_val = 0.
    for idx, m_op in trotterH.get_trotter_list():

        i,j = idx
        NR_ = min(NR,L1-i)
        NC_ = min(NC,L2-j)

        sub_tldm = tldm[i:i+NR_,j:j+NC_]

        bi = envIs[i][j:j+NC_]
        bo = envOs[i+NR_][j:j+NC_]

        bl = senvLs[i][j]
        br = senvRs[i][j+NC_] 

        exp_val = meas_obs(sub_tldm,trotterH.ops[m_op],ind0=trotterH.ind0[m_op],op_conn=trotterH.conn[m_op],
                           envs=[bl,bi,bo,br])
        
        obs_val += exp_val


    if return_norm:
        norm_val = np.sqrt( ENV_DM.ovlp_from_bound(envIs[L1]) )
        if np.isnan(norm_val):  print 'meas trotterH E nan'
        obs_val = obs_val/(norm_val**2)
        return obs_val, norm_val
    else:
        return obs_val



def meas_obs_mag(tldm,pauli_op='SZ',bounds=None,XMAX=100,return_norm=False,contract_SL=False,scaleX=1):
    # pepx could also be pepo ('dm','DM','rho','pepo')

    L1, L2 = tldm.shape
    NR, NC = (1,1)

    senvLs, envIs, envOs, senvRs = meas_get_bounds(tldm,bounds,(NR,NC),XMAX,contract_SL,scaleX)
        

    # get exp val
    op = Op.paulis[pauli_op]
    obs_val = 0.

    for idx in np.ndindex(L1,L2):

        i,j = idx

        sub_tldm = tldm[i:i+1,j:j+1]

        bi = envIs[i][j:j+1]
        bo = envOs[i+1][j:j+1]
        bl = senvLs[i][j]
        br = senvRs[i][j+1] 

        # meas obs 
        app_tldm = sub_tldm.copy()
        app_tldm[0,0] = np.einsum('liorda,Dd->liorDa',app_tldm[0,0],op)
        exp_val = ENV_DM.embed_sites_ovlp(np.conj(sub_tldm),app_tldm,[bl,bi,bo,br])
        
        obs_val += exp_val


    if return_norm:
        norm_val = np.sqrt( ENV_DM.ovlp_from_bound(envIs[L1]) )
        obs_val = obs_val/(norm_val**2)
        return obs_val, norm_val
    else:
        return obs_val


# def meas_obs_prod_op(tldm,ops,idxs,envs_list=None,XMAX=100,return_norm=False,contract_SL=False,scaleX=1):
#     ''' tldm:  (sub) tldm
#         ops:   list of operators (list of pauli key strings) to be applied at indices idxs
#         idxs:  list of list of indices that each operator (uxd) is applied to
#         envs_list:  bL, bI, bO, bR
#     '''
# 
#     app_tldm = tldm.copy()
#     for i in range(len(ops)):
#         try:                # ops[i] = 'SX', 'SY', 'SZ', or 'ID'
#             op_mat = Op.paulis[ops[i]]
#         except(KeyError):   # ops[i] should be an operator (2D matrix)
#             op_mat = ops[i]
# 
#         app_tldm[idxs[i]] = np.einsum('liorda,Dd->liorDa',app_tldm[idxs[i]],op_mat)
# 
#     if envs_list is None:
#         obs_val = vdot(tldm,app_tldm,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
#         if return_norm:  norm_val = norm(tldm,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
#     else: 
#         obs_val = ENV_DM.embed_site_ovlp(np.conj(tldm),app_tldm,envs_list,XMAX=XMAX,scaleX=scaleX)
#         if return_norm:  norm_val = ENV_DM.embed_site_norm(tldm,envs_list,XMAX=XMAX,scaleX=scaleX)
# 
#     obs_val = np.asscalar(obs_val)
#     if return_norm:
#         return obs_val/(norm_val**2), norm_val
#     else:
#         return obs_val


def meas_obs_prod_op(tldm,ops,idxs,bounds=None,XMAX=100,return_norm=False,contract_SL=False,scaleX=1):

    app_tldm = tldm.copy()
    min_r, max_r = 0,0 

    # bounds=None

    for i in range(len(ops)):

        if idxs[i][0] < min_r:   min_r = idxs[i][0]
        if idxs[i][0] > max_r:   max_r = idxs[i][0]

        try:                # ops[i] = 'SX', 'SY', 'SZ', or 'ID'
            op_mat = Op.paulis[ops[i]]
        except(KeyError):   # ops[i] should be an operator (2D matrix)
            op_mat = ops[i]

        app_tldm[idxs[i]] = np.einsum('liorda,Dd->liorDa',app_tldm[idxs[i]],op_mat)

    if bounds is None:
        obs_val = vdot(tldm,app_tldm,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
        if return_norm:  norm_val = norm(tldm,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
    else:
        senvLs, envIs, envOs, senvRs = bounds
        sbL = MPX.ones([(1,1)]*(max_r-min_r+1))
        sbR = MPX.ones([(1,1)]*(max_r-min_r+1))
        envs_list = [sbL, envIs[min_r], envOs[max_r+1], sbR]

        # print 'idxs', idxs
        # print [m.shape for m in sbL], [m.shape for m in envIs[min_r]]
        # print [m.shape for m in sbR], [m.shape for m in envOs[max_r+1]]
       
        obs_val = ENV_DM.get_sub_ovlp(np.conj(tldm[min_r:max_r+1,:]), app_tldm[min_r:max_r+1,:], envs_list,
                                   side='L', XMAX=XMAX)
        if return_norm:  norm_val = np.sqrt( ENV_DM.get_sub_ovlp(np.conj(tldm[min_r:max_r+1,:]),tldm[min_r:max_r+1,:],
                                                                 envs_list,side='L',XMAX=XMAX) )
        # obs_val = ENV_DM.embed_site_ovlp(np.conj(tldm),app_tldm,envs_list,XMAX=XMAX)
        # if return_norm:  norm_val = ENV_DM.embed_site_norm(tldm,envs_list,XMAX=XMAX)

    if return_norm:
        return obs_val/(norm_val**2), norm_val
    else:
        return obs_val


def meas_obs_corr(tldm,pauli_op=['SZ','SZ'],ref_ind=None,axis=None,bounds=None,XMAX=100,return_norm=False,
                  contract_SL=False,scaleX=1):
    # pepx could also be pepo ('dm','DM','rho','pepo')

    L1, L2 = tldm.shape

    op1 = Op.paulis[pauli_op[0]]
    op2 = Op.paulis[pauli_op[1]]
    obs_val = []

    if ref_ind is None:        i0,j0 = (L1/2,L2/2)
    else:                      i0,j0 = ref_ind

    if axis is None:
        if L1 < L2:     axis = 1
        else:           axis = 0

    if   axis == 0:     ir,jr = (L1,1)
    elif axis == 1:     ir,jr = (1,L2)
    else :              ir,jr = (L1,L2)
        

    for idx in np.ndindex(ir,jr):
        tldm_ = tldm.copy()

        i,j = idx
        tldm_[i0,j0] = np.einsum('liorda,Dd->liorDa',tldm_[i0,j0],op1)
        tldm_[i ,j ] = np.einsum('liorda,Dd->liorDa',tldm_[i ,j ],op2)

        exp_val = vdot(tldm,tldm_,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
        
        obs_val.append( exp_val )

    if return_norm:
    
        norm_val = np.sqrt( norm(tldm,contract_SL=contract_SL,scaleX=scaleX) )
        obs_val = [ov/(norm_val**2) for ov in obs_val]
        return obs_val, norm_val

    else:

        return obs_val

