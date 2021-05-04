import numpy as np
import scipy.linalg as linalg
import copy
import time

import tens_fcts as tf
import MPX
import Operator_1D as Op
import TimeEvolution as TE
import MPX_GSEsolver



def combineBonds(lmpx):
    # l x uo x do x di x ui x r --> l x (uo*do) x (di*ui) x r

    L = len(lmpx)

    new_sites = []
    for ii in range(L):
        assert(lmpx[ii].ndim > 3), 'lmpx has <1 physical bonds (assumes 2 virtual bonds)'
        new_sites.append(tf.reshape(lmpx[ii],range(1,lmpx[ii].ndim,2)))

    return MPX.MPX(new_sites)


def splitBonds(lmpx): 
    # l x (uo*do) x (di*ui) x r --> l x uo x do x di x ui x r 

    L = len(lmpx)

    new_sites = []
    for ii in range(L):
        s = lmpx[ii]
        ds = [int(d**0.5) for d in s.shape[1:-1] for x in range(2)]
        assert(np.all(np.array([ds[i]*ds[i+1] for i in range(0,len(ds),2)])
                        == np.array(s.shape[1:-1]))), 'error in splitting bonds'
        new_sites.append(s.reshape([s.shape[0]]+ds+[s.shape[-1]]))

    return MPX.MPX(new_sites)


#:::::::::::::::::::::::::::::::::::::::::::::::::
#          density matrix operations 
#:::::::::::::::::::::::::::::::::::::::::::::::::


def take_trace(mpdo,ind=0,side='R'):
    """
    assumes mpdo is MPO like (not MPS like)
    takes trace to the right/left of ind (inclusive), as determined by side
    to take trace of entire system:  ind = 0
    """

    sideR = (side == 'R' or side == 'B')
    sideL = (side == 'L' or side == 'A')
     
    # print mpdo.phys_bonds 
    # print mpdo[0].shape  
    if np.any([mpdo[i].ndim == 3 for i in range(len(mpdo))]):
        mpdo_ = splitBonds(mpdo)
        recombine =  True
    else:
        mpdo_ = mpdo.copy()
        recombine = False

    ind_ = ind   
    if ind_ == 0:     ind = 1
        # if ind = 0 first do calc as if ind = 1 (trace over remaining site at end)


    if side == 'R' or side == 'B':
        toKeep  = mpdo_[:ind]      # sites to keep
        toTrace = mpdo_[ind:]      # sites to trace over
    elif side == 'L' or side == 'A':
        toKeep  = mpdo_[:ind][::-1]
        toTrace = mpdo_[ind:][::-1]
    else:
        raise ValueError, 'choose side A/L or B/R'


    traced = toKeep[-1].copy()
    for i in range(len(toTrace)):
        t = toTrace[i]
        if t.ndim == 6:
            traceString = 'bcdcde'
        elif t.ndim == 4:
            # bonds structure  == [1,2,0,1] or [1,0,2,1]:or [1,1,1,1]
            traceString = 'bcce'
        else:
            raise NotImplementedError, 'have not yet implemented trace for site of ndim %d'%t.ndim
 
        # print 'traced over', np.einsum('bcce',t)
        traced = np.einsum('...b,'+traceString+'->...e',traced,t)
        # print traced.squeeze()

    toKeep[-1] = traced.copy()
    if side == 'L' or side == 'A':  toKeep.reverse()
    if recombine:                   toKeep = combineBonds(toKeep)   # put in original form

    if ind_ == 0:                    # originally wanted trace over entire system
        assert((traced.shape[0],traced.shape[-1]) == (1,1)),traced.shape
        if   traced.ndim == 4:
            traced = np.einsum('aiia->',traced)
        elif traced.ndim == 6:
            traced = np.einsum('aijija->',traced)
            print 'did 4 bond trace'
        else:
            print 'trace err', traced.ndim
        
        toKeep = traced

    return toKeep


def normalize(mpdo, target_norm=1.0):
    norm = take_trace(mpdo)
    return MPX.mul(target_norm/norm,mpdo)


def measMPO(mpdo,pos0,numSites,MPO):

    if np.any([m.ndim==3 for m in mpdo]):
        new_mpdo = splitBonds(mpdo)
    else:
        new_mpdo = mpdo.copy()

    mpdo_mpo = new_mpdo[pos0:pos0+numSites].dot(MPO)
    new_mpdo[pos0:pos0+numSites] = mpdo_mpo
    expVal   = take_trace(new_mpdo,0)

    return expVal   # scalar or tensor to contract input site with



def get_eigvals(mpdo,get_eigvecs=False,assume_hermitian=False):
    ''' do exact diagonalization to get eigenvalues of DM '''

    assert(len(mpdo)<6),'use smaller mpdo for ED'

    if np.any([mpdo[i].ndim == 3 for i in range(len(mpdo))]):
        mpdo_ = splitBonds(mpdo)

    L = len(mpdo_)
    rho = mpdo_.getSites()
    axT = tf.site2io(L)
    sqdim = int(np.sqrt(np.prod(rho.shape)))

    rho = rho.transpose(axT).reshape(sqdim,sqdim)

    # if assume_hermitian:
    #     assert(np.linalg.norm(rho - np.conj(rho.T)) < 1.0e-8), 'rho is not hermitian'

    if get_eigvecs:

        if assume_hermitian:
            evals, evecs = np.linalg.eigh(rho)
        else:
            evals, evecs = np.linalg.eig(rho)
        return evals, evecs

    else:
        if assume_hermitian:
            evals = np.linalg.eigvalsh(rho)
        else:
            evals = np.linalg.eigvals(rho)
        return evals
 

def get_matrix(mpdo):

    if np.any([mpdo[i].ndim == 3 for i in range(len(mpdo))]):
        mpdo_ = splitBonds(mpdo)

    L = len(mpdo_)
    rho = mpdo_.getSites()
    axT = tf.site2io(L)

    sqdim = int(np.sqrt(np.prod(rho.shape)))

    rho = rho.transpose(axT).reshape(sqdim,sqdim)

    return rho

     


def meas_fidelity(mpdo1,mpdo2):
    ''' mpdo are density matrices'''
    ''' quantum fidelity between two density matrices '''

    L = len(mpdo1)
    rho1 = mpdo1.getSites()
    rho2 = mpdo2.getSites()
    axT = tf.site2io(L)
    sqdim = int(np.sqrt(np.prod(rho1.shape)))

    rho1 = rho1.transpose(axT).reshape(sqdim,sqdim)
    rho2 = rho2.transpose(axT).reshape(sqdim,sqdim)
    

    ## check if mpdo1, mpdo2 are diagonal
    rho1d = np.diag(np.diag(rho1))
    rho2d = np.diag(np.diag(rho2))

    is_diag1 = np.linalg.norm(rho1d-rho1) < 1.0e-12
    is_diag2 = np.linalg.norm(rho2d-rho2) < 1.0e-12

    if is_diag1 and is_diag2:     # both are diagonal
        fidelity = np.trace(np.sqrt(np.matmul(rho1,rho2)))**2
    elif is_diag1:                # rho1 is diagonal, but rho2 is not    
        rho1_sqrt = np.sqrt(rho1)
        temp = np.matmul(rho1_sqrt,np.matmul(rho2,rho1_sqrt))
        fidelity = np.trace(linalg.sqrtm(temp))**2
    elif is_diag2:                # rho2 is diagonal, but rho1 is not
        rho2_sqrt = np.sqrt(rho2)
        temp = np.matmul(rho2_sqrt,np.matmul(rho1,rho2_sqrt))
        fidelity = np.trace(linalg.sqrtm(temp))**2
    else:
        rho1_sqrt = linalg.sqrtm(rho1)
        temp = np.matmul(rho1_sqrt,np.matmul(rho2,rho1_sqrt))
        fidelity = np.trace(linalg.sqrtm(temp))**2
        
    return fidelity


#:::::::::::::::::::::::::::::::::::::::::::::::::
#               MPDO time evolution 
#:::::::::::::::::::::::::::::::::::::::::::::::::

def Hilbert_TE(mpdo,H_MPO,dt,t,te_method,order=4,compress=True, DMAX=100):
    """
    perform time evolution on density matrix in Hilbert space
    """
    
    errs_t = []
    mpdo_t = []

    t_ = 0
    errs = np.array([0.]*(len(mpdo)-1))

    rho = mpdo.copy()
    while t_ < abs(t):
        time1 = time.time()
        rho, e1 = TE.timeEvolve(rho, H_MPO, dt, dt, te_method, order, compress, DMAX, direction=0)
        time2 = time.time()
        rho = MPX.transposeUD(rho[0])
        errs += e1[0]
        # print 'TE', time.time()-time1
        
        time3 = time.time()
        # rho, e2 = TE.timeEvolve(rho, H_MPO, -dt, dt, te_method, order, compress, DMAX, direction=1)
        rho, e2 = TE.timeEvolve(rho, H_MPO, -np.conj(dt), dt, te_method, order, compress, DMAX, direction=1)
        rho = MPX.transposeUD(rho[0])
        errs += e2[0]
        # print 'TE_2', time.time()-time3

        # print 'H_TE 1', [s.shape[0] for s in rho]
        # rho, e3 = MPX.compress(rho,DMAX)   # additional compression reduces bond dimension
        # errs += e3[0]
        # print 'H_TE 2', [s.shape[0] for s in rho]

        errs_t.append(errs)
        mpdo_t.append(rho)
 
        t_ += abs(dt)

    return mpdo_t, errs_t


def Liouv_TE(mpdo,L_MPO,dt,t,te_method,order=4,compress=True,DMAX=100, direction=0,normalize=True,norm=1.0):
    """
    perform time evolution on density matrix in Liouv space

    L_MPO:  already exponetiated exactly or via taylor expansion
    """

    # if mpdo[0].ndim == 3 and L_MPO[0].ndim == 4:
    #     do_reshape = False
    #     rho = mpdo
    #     op  = L_MPO
    # elif mpdo[0].ndim == 4 and L_MPO[0].ndim == 6:
    #     do_reshape = False
    #     rho = combineBonds(mpdo)
    #     op  = combineBonds(L_MPO)
    # else:
    #     raise ValueError, 'mpdo and/or L_MPO of wrong/mismatched dimension'

    rho = mpdo
    op  = L_MPO
    do_reshape = False

    errs_t = []
    mpdo_t = []

    nSteps = 0
    totSteps = int(np.abs(t/dt))
    errs = np.array([0.]*(len(mpdo)-1))

    dir2 = direction
    while nSteps < totSteps:
        # rho, e1 = TE.timeStep(rho, op, dt, te_method, order)

        # print 'LTE', dir2
        # print type(rho)
        # rho_t, e_t = TE.timeEvolve(rho, op, dt, dt, te_method, order, compress, DMAX, direction, normalize)
        rho_t, e_t = TE.timeEvolve(rho, op, dt, dt, te_method, order, compress, DMAX, dir2, 
                                   normalize, mps_type='rho', norm=norm)
        rho = rho_t[0]
        # print [s.shape[0] for s in rho]
        # rho = MPX.compress(rho,-1)[0]
        e1  = e_t[0]
        # print type(rho)

        errs_t.append(errs[-1]+e1)
        mpdo_t.append(rho)
 
        dir2 = (direction+nSteps*order)%2   # so can set direction if t = dt, and can still change with time
             # switches if timeEvolve uses odd number of steps
        # print [s.shape[0] for s in rho]

        nSteps += 1

    return mpdo_t, errs_t
    


def getLiouvillian(H_Op):
    ## H x I - I x H  --> MPOlist with physical bonds d**2

    if isinstance(H_Op, MPX.MPX):
        H_MPO = H_Op
        ds = [d[0] for d in H_Op.phys_bonds]
    elif isinstance(H_Op, Op.MPO):
        H_MPO = H_Op.getMPX()
        ds = H_Op.ds
    else:
        print 'getLiouvillian:  H_Op needs to be Op.MPO or MPX'
        exit()

    L1 = []
    L2 = []

    ## construct L1
    for i in range(len(H_MPO)):
        iden = np.eye(ds[i])
        # l x uo x di x r, ui x do  --> uo x do x di x ui
        L1.append( np.einsum('labr,ij->lajbir',H_MPO[i],iden) )   # MPOlist applied to up DM bonds 
    
    ## construct L2
    for i in range(len(H_MPO)):
        iden = np.eye(ds[i])
        L2.append( np.einsum('ab,lijr->lajbir',iden,H_MPO[i]) )   # MPOlist applied to down DM bonds

    ## construct L = L1 - L2

    L1 = combineBonds(MPX.MPX(L1))
    L2 = combineBonds(MPX.MPX(L2))

    Liouv = L1 - L2 

    return Liouv


def createThermalState(H,beta,U=None,D=100,te_method='Taylor',order=4):
    """
    returns exp(-beta H) = exp(-beta H)/2 UU.T exp(-beta H)/2 via imaginary time propagation
   
    parameters:
    ----------
    U:       unitary, such as the identity MPO
    beta:    1/kb T
    dbeta:   "time step" used to reach beta

    """
    
    if U is None:   rho = MPX.eye(H.ds)
    else:           rho = U.dot(MPX.transposeUD(U.conj()))

    H_MPO = H.getMPX()

    if te_method == 'Taylor':
         max_eval = Op.mpo_targetEVal(H_MPO)
         dt = min(2.785/max_eval * 0.95, beta*1./10.)
         TE_mpo = TE.Taylor_MPO(H_MPO,-1.j*dt,order=4)
    elif te_method == 'exact':
         if not(beta==0.):  dt = beta
         else:              dt = 0.1  # arbitrary
         TE_mpo = TE.exact_exp(H_MPO,1.j*dt)
    else:
         print 'choose valid te_method'
         exit()

    if not(beta==0.):
         mpdo_t, errs_t = TE.timeEvolve(rho, TE_mpo, -1j*dt, beta, te_method, order=order, DMAX=D,
                                        normalize=False, mps_type='rho')
    else:
         mpdo_t = [rho]

    th_state = mpdo_t[-1]
    th_state = MPX.mul(1./take_trace(th_state,0),th_state)

    return th_state


def createProductThermalState(op_list,beta,U=None):
    """
    returns exp(-beta H) = exp(-beta H)/2 UU.T exp(-beta H)/2 via imaginary time propagation
   
    parameters:
    ----------
    U:       unitary, such as the identity MPO
    beta:    1/kb T
    dbeta:   "time step" used to reach beta
    op_list:  list of [dxd] matrices corresponding to on site terms in H

    """

    ds = [m.shape[0] for m in op_list]
    
    if U is None:   rho = MPX.eye(ds)
    else:           rho = U.dot(MPX.transposeUD(U.conj()))

    if np.isinf(beta):
        th_state = MPX.MPX( [ np.diag([1.]+[0.]*(d-1)).reshape(1,d,d,1) for d in ds ] )
    else:
        expH = TE.expH_product_mpo(op_list, beta*-1.j)  # product MPO

        th_state = MPX.dot( expH, rho )
        th_state = MPX.mul(1./take_trace(th_state),th_state)


    return th_state


# def disentangler(two_sites, initU=None, max_it=50, tol=tol):
#     """
#     Determines unitary to disentangle two sites
# 
#     parameters:
#     -----------
#     two_sites:  [MPX]     sites to be disentangled (1 x d1 x d1 x k, k x d2 x d2 x 1)
#     initU:      [ndarray] starting point of iterative solver.  default is identity
#     max_it:     maximum number of iterations before giving up
#     tol:        convergence tolerance
# 
#     """
#      
#     bonds_0 = [d[0] for d in two_sites.phys_bonds]
#     if initU is None:    U = np.eye(np.prod(bonds_0))
#     else:                U = initU.copy()
# 
#     rho = np.einsum('aijb,bklc->aikjlc',two_sites[0], two_sites[1])   # in 'io' ordering
#     rho = rho.reshape(np.prod(bonds_0),np.prod(bonds_0))   # square matrix (d1*d2) x (d1*d2)
# 
#     it = 0   
#     conv = False
#     while it < max_it or not conv:
#         env = np.dot( rho, np.dot(U, rho) )
#         u,s,v = np.linalg.svd(env)    # 1xd1xd2 x r,  r x d1xd2x1
# 
#         U_ = np.dot(v.T,u.T)       # r x r (but r=d1xd2 bc using full matrices)
#         conv = np.allclose(U, U_)
#         U =  U_
# 
#         it += 1
# 
#     assert( np.allclose(np.dot(U,U.T), np.eye(np.prod(bonds_0))) ), 'U not unitary'
#     U = U.reshape((1,)+tuple(bonds_0)+tuple(bonds_0)+(1,)) 
#     axT = tf.io2site(2)
#     print it
#     print U
# 
#     return U.transpose(axT)
    

def disentangler(two_sites):
    rho = np.einsum('aijb,bklc->aikjlc',two_sites[0], two_sites[1])   # output in 'io' ordering
    rho = tf.reshape(rho,'iii,iii')

    u,s,vt = np.linalg.svd(rho)   # u, v are d**2 x d**2 orthogonal matrices

    row_reorder_vt = [0, 2, 3, 1]
    
    u_  = u[:,:]
    vt_ = v[row_reorder_vt,:]

    U = np.einsum('ia,jb->ijab',u_,vt_)

    # new_rho = np.einsum('ijab,labr->lijr')




