import numpy as np
import copy
import time
import tens_fcts as tf

import Operator_2D as Op
import PEPS_env as penv
import PEP0_env as penv0
import PEPS_GL_env_nolam as ENV_GL
import PEPS_GL_env_nolam_SL as ENV_GLS
import PEPX

import PEPX_GL_trotterTE as TE


class PEPX_GL(PEPX.PEPX):
    """
    Lightweight PEPS/PEPO class in Gamma/Lambda decomposition

    2D array storing Gamma's
    2D array storing Lambda's
    2D array storing phys_bonds (same as before) (is it really necessary..?)

    """

    def __new__(cls,tparray,lambdas=None,phys_bonds=None):
        """
        Parameters
        ----------
        tparray [array_like]:  nested list of tensors; limited to square lattice
        lambdas [array_like];  2D array of vectors repping diag matrices connecting each bond
        phys_bonds [2D array of tuples]:  physical bond dimension of each lattice site
        """

        # site tensors
        if isinstance(tparray,np.ndarray) or isinstance(tparray,cls):
            pepx = tparray.view(cls)   # pepx is an ndarray of ndarrays
        else:
            ## pepx is a nested list of ndarrays
            # unpack list, assuming 2D
            
            L1, L2 = len(tparray), len(tparray[0])
            pepx = np.empty((L1,L2),dtype=object)
            for i,j in np.ndindex((L1,L2)):
                pepx[i,j] = tparray[i][j]
            pepx = pepx.view(cls)

        # if not pepx.ndim == 2:   raise TypeError, 'peps array must be 2D not %d' %pepx.ndim


        # lambdas:  a 3D array, where 3rd dimension are l,i,o,r singular vals (1D array)
        # defined such that elements corresponding to same bond reference the same object
        # so that when that object is updated, both references are updated

        # if isinstance(tparray,cls):
        #     pepx.lambdas = tparray.lambdas
        try:
            pepx.lambdas = tparray.lambdas
        except(AttributeError):
            if isinstance(lambdas,np.ndarray):   # dictionary of 1D arrays
                pepx.lambdas = lambdas.copy()
            else:
                L1,L2 = pepx.shape
                lambdas = np.empty((L1,L2,4),dtype=object)

                # body + edges
                for idx in np.ndindex(L1,L2):
                    Dl,Di,Do,Dr = tparray[idx].shape[:4]
                    lambdas[idx] = [np.ones(Dl),np.ones(Di),np.ones(Do),np.ones(Dr)]

                pepx.lambdas = lambdas
           

        # physical bonds
        if phys_bonds is None:
            pepx.phys_bonds = np.empty(pepx.shape,dtype=tuple)
            for idx, x in np.ndenumerate(pepx):
                pepx.phys_bonds[idx] = x.shape[4:]
        else:
            # is an ndarray of tuples already
            try:
                if isinstance(phys_bonds[0,0],tuple):
                    pepx.phys_bonds = phys_bonds.copy()
                else: 
                    raise (TypeError)
            except(TypeError):     # is a list / not an an array of object type
                pepx.phys_bonds = np.empty(pepx.shape,dtype=tuple)
                for idx, x in np.ndenumerate(pepx):
                    try:
                        pepx.phys_bonds[idx] = tuple( phys_bonds[idx[0]][idx[1]] )
                    except (TypeError):    # d is an int
                        pepx.phys_bonds[idx] = (phys_bonds[idx[0]][idx[1]],)

        return pepx


    def __array_finalize__(self,pepx):
        if pepx is None:  return
        self.lambdas = getattr(pepx, 'lambdas',None)
        self.phys_bonds = getattr(pepx, 'phys_bonds',None)  


    def __getitem__(self,item):
        sliced_pepx = super(PEPX_GL,self).__getitem__(item)
        try:
            sliced_pepx.phys_bonds = self.phys_bonds.__getitem__(item)
            sliced_pepx.lambdas = self.lambdas.__getitem__(item)
        except(AttributeError):   
            # np.ndarray has no attribute phys_bonds.  occurs when trying to print? i guess it's called recursively
            pass
        return sliced_pepx

    def __setitem__(self,item,y):
        super(PEPX_GL,self).__setitem__(item,y)
        try:                      # y is also an MPX object
            self.phys_bonds.__setitem__(item,y.phys_bonds)
            self.lambdas.__setitem__(item,y.lambdas)
        except(AttributeError):   # y is just an ndarray
            # print 'setting item'
            # print 'self',self.phys_bonds

            try:
                temp_bonds = y.shape[4:]      # y is an ndarray tens
            except(AttributeError):
                temp_bonds = [yy.shape for yy in y]   # y is a list of ndarray tens

            self.phys_bonds.__setitem__(item,temp_bonds)


    def __add__(pepx1, pepx2):
        return add(pepx1, pepx2)
    
    def __sub__(pepx1, pepx2):
        return add(pepx1, mul(-1,pepx2))

    def __mul__(pepx, alpha):
        return mul(alpha, pepx)

    def __neg__(pepx):
        return pepx.mul(-1)
    
    def dot(pepx1, pepx2):
        return dot(pepx2,pepx1)

    def norm(pepx):
        return norm(pepx)

    def outer(pepx1,pepx2):
        return outer(pepx1,pepx2)

    def transposeUD(pepo):
        return PEPX.transposeUD(pepo)

    def copy(pepx):
        return copy(pepx)



#########################################
#########   creation fcts   #############
#########################################

##### functions to create PEPX ######

def get_product_lambdas(Ls):
    ''' create 2D array lambda for uniform D state where all lambdas are ones
        otherwise just use fct in definition
    '''

    L1,L2 = Ls
    lambdas = np.empty(Ls+(2,),dtype=object)

    # body
    for idx in np.ndindex(L1,L2):
        lambdas[idx] = [np.ones(1),np.ones(1),np.ones(1),np.ones(1)]

    return lambdas    


def create(dp, D, lambdas=None, fn=np.zeros):
    """
    Create random MPX object as ndarray of ndarrays

    Parameters
    ----------
    dp : nested list of ints or list of 2-tuples
      Specifies physical dimensions of PEPS or PEPO. 
    D : int, maximum bond dimension

    Returns
    -------
    pepx : MPX object, ndarray of ndarrays
       PEPS or PEPO
    """

    L1, L2 = len(dp), len(dp[0])

    pepx = np.empty((L1,L2), dtype=np.object)

    # fill in MPX with random arrays of the correct shape   
    for i in range(L1):
        for j in range(L2):
            if i == 0:
               if j == 0:       pepx[i,j] = fn((1,1,D,D)+dp[i][j])   # tensors are lior(ud)
               elif j == L2-1:  pepx[i,j] = fn((D,1,D,1)+dp[i][j])
               else:            pepx[i,j] = fn((D,1,D,D)+dp[i][j])
            elif i == L1-1:
               if j == 0:       pepx[i,j] = fn((1,D,1,D)+dp[i][j]) 
               elif j == L2-1:  pepx[i,j] = fn((D,D,1,1)+dp[i][j])
               else:            pepx[i,j] = fn((D,D,1,D)+dp[i][j])

            elif j == 0:        pepx[i,j] = fn((1,D,D,D)+dp[i][j])
            elif j == L2-1:     pepx[i,j] = fn((D,D,D,1)+dp[i][j])

            else:               pepx[i,j] = fn((D,D,D,D)+dp[i][j])

    return PEPX_GL(pepx, lambdas=lambdas, phys_bonds=dp)   ## default lambdas

def empty(dp, D=1):
    return create(dp, D, fn=np.empty)

def zeros(dp, D=1):
    return create(dp, D, fn=np.zeros)

def ones(dp, D=1):
    return create(dp, D, fn=np.ones)

def rand(dp,D=1, seed = None):
    if seed is not None:
        np.random.seed(seed)
    return create(dp, D, fn=np.random.random)

def normalized_rand(dp, D=1, seed=None):
    pepx = rand(dp,D,seed)
    norm = pepx.norm()
    return mul(1./norm,pepx)


def random_product_state(dp):
    '''ensures that all tensors have norm of 1'''
    
    pepx = empty(dp,1)

    for idx in np.ndindex(pepx.shape):
        dp  = pepx.phys_bonds[idx]
        dp_ = np.prod(dp)
        site = np.random.rand( dp_ ).reshape((1,1,1,1,)+dp)
        pepx[idx] = site / np.linalg.norm(site)

    print 'random product state'
    print [m for idx, m in np.ndenumerate(pepx)]

    return pepx


def random_tens(shape_array,normalize=True):
    ''' define random peps of tensors with shapes specified in shape_array '''

    L1,L2 = shape_array.shape
    pepx = np.empty((L1,L2))

    for idx in np.ndindex((L1,L2)):
        pepx[idx] = np.random.random.random_sample(shape_array[idx])

    pepx_ = PEPX_GL(pepx)

    if normalize:
        pepx_norm = norm(pepx_)
        pepx_ = mul(1./pepx_norm, pepx_)

    return pepx_


def product_peps(dp, occ):
    """
    generate PEPS product state

    Parameters
    ----------
    dp:   list/array of integers specifying dimension of physical bonds at each site
    occ:  occupancy vector (len L), numbers specifying physical bond index occupied

    Returns
    -------
    returns product state mps according to occ
    """

    L1,L2 = len(dp),len(dp[0])
    peps  = zeros(dp, 1)
    for i in range(L1):
        for j in range(L2):
            peps[i,j][0,0,0,0,occ[i][j]] = 1.

    return peps


def product_pepo(dp,occ):
    """ generate PEPO product state """

    L1,L2 = len(dp),len(dp[0])
    pepo  = zeros(dp, 1)
    for i in range(L1):
        for j in range(L2):
            x, y = occ[i][j]
            pepo[i,j][0,0,0,0,x,y] = 1.

    return pepo


def product_kvec(Ls,kvec,pepx_type='peps'):

    occs = np.empty(Ls,dtype=np.int)

    for i,j in np.ndindex(Ls):
        occs[i,j] = int((i%(kvec[0]+1) + j%(kvec[1]+1))%2)

    print 'product kvec occs', occs

    if pepx_type == 'peps':
        return product_peps([[(2,)]*Ls[1]]*Ls[0],occs)
    elif pepx_type == 'pepo':
        return product_pepo([[(2,2)]*Ls[1]]*Ls[0],occs)


# def eye(dp):
#     """
#     generate identity MPO
#     dp:  list of physical bonds (not tuples) at each site
#     """
# 
#     if not isinstance(dp, np.ndarray):  dp  = np.array(dp)
#     L1,L2 = dp.shape
# 
#     id_pepo = empty(dp,1)
#     for i in range(L1):
#         for j in range(L2):
#             id_pepo[i,j] = np.eye(dp[i,j]).reshape(1,1,1,1,dp[i,j],dp[i,j])
# 
#     return id_pepo

def eye(dp):
   ''' dp are phys bonds (tuples) at each site '''

   id_pepo = empty(dp,1)
   L1,L2 = id_pepo.shape
   for i in range(L1):
       for j in range(L2):
           dphys = dp[i][j][0]
           id_pepo[i,j] = np.eye(dphys).reshape(1,1,1,1,dphys,dphys)

   return id_pepo
    


def copy(pepx):
    # deeper copy than mpx.copy, as it copies the ndarrays in mpx
    new_pepx = empty(pepx.phys_bonds,1)
    for idx, tens in np.ndenumerate(pepx):
        new_pepx[idx] = tens.copy()
        new_pepx.lambdas[idx] = [m.copy() for m in pepx.lambdas[idx]]
    return new_pepx


def transpose_pepx(pepx):

    new_pepx = empty(pepx.phys_bonds.T,1)
    for idx, tens in np.ndenumerate(pepx):
        i,j = idx
        try:      new_pepx[j,i] = tens.transpose([1,0,3,2,4])
        except:   new_pepx[j,i] = tens.transpose([1,0,3,2,4,5])
        new_pepx.lambdas[j,i] = pepx.lambdas[i,j][[1,0,3,2]]
    return new_pepx


#############################
####### other fcts? #########
#############################

def element(pepx, occ):
    """
    returns value of in mpx for given occupation vector
    """
    if not isinstance(occ,np.ndarray):  occ = np.array(occ)   
    L1, L2 = occ.shape

    mats = np.array( occ.shape )
    
    if len(occ[0]==2):
        for ind in np.ndindex(occ):
            mats[ind] = pepx[ind][:,:,:,:,occ[ind][0],occ[ind][1]]
    else:
        for ind in np.ndindex(occ):
            mats[ind] = pepx[ind][:,:,:,:,occ[ind]]
    
    return penv0.contract(mats)


def conj_transpose(pepx):

    if pepx[0,0].ndim == 5:  
        pepx_cc = np.conj(pepx.view(np.ndarray))
        pepx_ = PEPX_GL(pepx_cc,pepx.lambdas,pepx.phys_bonds)
    elif pepx[0,0].ndim == 6:  
        pepx_cc = np.conj(pepx,view(np.ndarray))
        pepx_ = PEPX_GL(pepx_cc,pepx.lambdas,pepx.phys_bonds)
        for i in np.ndenumerate(pepx_):
            pepx_[i] = pepx_[i].transpose([0,1,2,3,5,4])
    return pepx_


def flatten(pepx):
    """   Converts PEPO object into PEPS    """

    peps = np.empty(pepx.shape,dtype=object)
    for ind in np.ndindex(pepx.shape):
        ptens = pepx[ind]
        if ptens.ndim == 5:  peps[ind] = ptens
        else:                peps[ind] = tf.reshape(ptens,'i,i,i,i,...',group_ellipsis=True)
    return PEPX_GL(peps,pepx.lambdas)


def unflatten(pepx,dbs):
    '''  converts PEPS object into PEPO '''

    pepo = np.empty(pepx.shape,dtype=object)
    for ind in np.ndindex(pepx.shape):
        ptens = pepx[ind]
        pepo[ind] = ptens.reshape( ptens.shape[:4]+dbs[ind] )
    return PEPX_GL(pepo,pepx.lambdas,dbs)


def transposeUD(pepx):
    pepx_ = pepx.copy()
    if np.all( [pepx[idx].ndim == 5 for idx in np.ndindex(pepx.shape)] ):
        return pepx_
    else:
        for ind in np.ndindex(pepx_.shape):
            try:  pepx_[ind] = pepx_[ind].transpose((0,1,2,3,5,4))
            except(ValueError):
                pepx_[ind] = pepx[ind]
        return pepx_


def compress(pepx,DMAX):

    if DMAX > 0:
        pepx_ = empty(pepx.phys_bonds)
        for idx in np.ndindex(pepx.shape):
            pepx_[idx] = pepx[idx][:DMAX,:DMAX,:DMAX,:DMAX]
            pepx_.lambdas[idx][0] = pepx.lambdas[idx][0][:DMAX]
            pepx_.lambdas[idx][1] = pepx.lambdas[idx][1][:DMAX]
            pepx_.lambdas[idx][2] = pepx.lambdas[idx][2][:DMAX]
            pepx_.lambdas[idx][3] = pepx.lambdas[idx][3][:DMAX]
    else:
        pepx_ = pepx.copy()

    return pepx_


def regularize(pepx):
    pepx_ = pepx.copy()

    L1,L2 = pepx.shape

    # vertical bonds
    for j in range(L2):
        for i in range(L1-1):

            v_bond = pepx.lambdas[i,j,2]
            normB = np.linalg.norm(v_bond)

            pepx_[i,j]   = pepx_[i,j]*(normB**0.5)
            pepx_[i+1,j] = pepx_[i+1,j]*(normB**0.5)
            pepx_ = set_bond(pepx_,[i,j],2,v_bond/normB)
        
    # horizontal bonds
    for i in range(L1):
        for j in range(L2-1):

            h_bond = pepx.lambdas[i,j,3]
            normB = np.linalg.norm(h_bond)

            pepx_[i,j]   = pepx_[i,j]*(normB**0.5)
            pepx_[i,j+1] = pepx_[i,j+1]*(normB**0.5)
            pepx_ = set_bond(pepx_,[i,j],3,h_bond/normB)

    return pepx_

    
##############################
#### arithemtic functions ####
##############################

def vdot(pepx1, pepx2, side='I',XMAX=100, contract_SL=False,scaleX=1):
    """
    vdot of two PEPS <psi1|psi2> (or PEPO-->PEPS). returns scalar

    cf. np.vdot
    """
    ovlp =  peps_dot(np.conj(flatten(pepx1)),flatten(pepx2),side=side,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
    return np.squeeze(ovlp)


def peps_dot(peps1, peps2, side='I',XMAX=100,contract_SL=False,scaleX=1):
    """
    dot of two PEPS, returns scalar
    """
    if contract_SL:
        ovlp = ENV_GLS.get_ovlp(peps1,peps2,side=side,XMAX=XMAX,scaleX=scaleX)
    else:
        ovlp = ENV_GL.get_ovlp(peps1,peps2,side=side,XMAX=XMAX)

    return ovlp


def norm(pepx,side='I',XMAX=100,contract_SL=False,scaleX=1): 
    """
    2nd norm of a MPX
    """
     
    norm_tens = peps_dot(np.conj(flatten(pepx)),flatten(pepx),side=side,XMAX=XMAX,contract_SL=contract_SL,
                         scaleX=scaleX)

    # # catch cases when norm is ~0 but in reality is a small negative number
    # assert(np.abs(np.imag(norm_val)/(np.real(norm_val)+1.0e-12)) < 1.0e-12), norm_val
    # assert(np.real(norm_val) > -1.0e-10), norm_val

    # return np.sqrt(np.abs(norm_val))
    norm_val = np.einsum('ee->',norm_tens)
    return np.sqrt(norm_val)


def trace_norm(pepx_gl,side='I',XMAX=100,contract_SL=False,scaleX=1):

    pepx = get_pepx(pepx_gl)
    return PEPX.trace_norm(pepx, side, XMAX, contract_SL,scaleX=scaleX)


def mul(alpha, pepx):
    """
    scales mpx by constant factor alpha (can be complex)
    """

    L1,L2 = pepx.shape
    new_pepx = pepx.copy()

    dtype = np.result_type(alpha,pepx[0,0])

    # # lambdas should be real (sing vals are all real)
    # const = np.abs(alpha)**(1./(2*L1*L2-L1-L2))
    # for idx in np.ndindex(L1,L2):
    #     if not (idx[0] == L1-1):   # not last row
    #         o_bond = pepx.lambdas[idx][2]
    #         new_pepx = set_bond(new_pepx,idx,2,const*o_bond)
    #     if not (idx[1] == L2-1):   # not right col
    #         r_bond = pepx.lambdas[idx][3]
    #         new_pepx = set_bond(new_pepx,idx,3,const*r_bond)

    const = np.abs(alpha)**(1./L1/L2)
    for idx in np.ndindex(L1,L2):
        new_pepx[idx] = const*new_pepx[idx]

    # change sign as specified by alpha
    if dtype == int or dtype == float:
        phase = np.sign(alpha)
    elif dtype == complex:
        phase = np.exp(1j*np.angle(alpha))
    else:
        raise(TypeError), 'not valid datatype %s' %dtype

    new_pepx[0,0] = np.array(new_pepx[0,0],dtype=dtype)*phase

    return new_pepx



def add(pepx1, pepx2, obc=(True,True,True,True)):
    """
    Direct sum of MPX's of the same shape
    obc:  if True, reshape MPX s.t. end dims are 1
    """
    L1, L2 = pepx1.shape

    assert pepx1.shape==pepx2.shape,\
           'add error: need to have same shapes: (%d,%d)'%(len(mpx1),len(mpx2))
    assert np.all(pepx1.phys_bonds==pepx2.phys_bonds),\
           'add error: need to have same physical bond dimensions'

    new_pepx = np.empty(pepx1.shape, dtype=np.object)
    new_lams = np.empty(pepx1.lambdas.shape, dtype=np.object)
    dtype = np.result_type(pepx1[0,0], pepx2[0,0])

    for idx in np.ndindex((L1,L2)):
        sh1 = pepx1[idx].shape
        sh2 = pepx2[idx].shape

        l1,u1,d1,r1 = sh1[:4]
        l2,u2,d2,r2 = sh2[:4]
        dp_sh = sh1[4:]

        new_site = np.zeros((l1+l2,u1+u2,d1+d2,r1+r2)+dp_sh,dtype=dtype)
        new_site[:l1,:u1,:d1,:r1] = pepx1[idx]
        new_site[l1:,u1:,d1:,r1:] = pepx2[idx]

        new_pepx[idx] = new_site.copy()
        for x in range(len(new_lams[idx])):
            i,j = idx
            new_lams[i,j,x] = np.append( pepx1.lambdas[i,j,x], pepx2.lambdas[i,j,x] )

    new_pepx = PEPX_GL(new_pepx,new_lams)

    if obc[0]:   # left boundary
        for i in range(L1):
            l1 = pepx1[i,0].shape[0]
            l2 = pepx2[i,0].shape[0]
 
            if l1 == l2:
                bondL = get_bond(new_pepx,(i,0),0)
                tens1 = apply_bond(new_pepx[i,0][:l1,:,:,:],bondL[:l1],0)
                tens2 = apply_bond(new_pepx[i,0][l1:,:,:,:],bondL[l1:],0)
                new_site = tens1 + tens2
                new_bond = (bondL[:l1] + bondL[l1:])/2.
                new_site = apply_inverse_bond(new_site,new_bond,0)
                new_pepx[i,0] = new_site.copy()
                new_lams[i,0,0] = new_bond

    if obc[1]:   # upper boundary
        for i in range(L2):
            i1 = pepx1[0,i].shape[1]
            i2 = pepx2[0,i].shape[1]
          
            if i1 == i2:  
                bondI = get_bond(new_pepx,(0,i),0)
                tens1 = apply_bond(new_pepx[0,i][:,:i1,:,:],bondI[:i1],1)
                tens2 = apply_bond(new_pepx[0,i][:,i1:,:,:],bondI[i1:],1)
                new_site = tens1 + tens2
                new_bond = (bondI[:i1] + bondI[i1:])/2.
                new_site = apply_inverse_bond(new_site,new_bond,1)
                new_pepx[0,i] = new_site.copy()
                new_lams[0,i,1] = new_bond

    if obc[2]:   # lower boundary
        for i in range(L2):
            o1 = pepx1[-1,i].shape[2]
            o2 = pepx2[-1,i].shape[2]
          
            if o1 == o2:  
                bondO = get_bond(new_pepx,(-1,i),0)
                tens1 = apply_bond(new_pepx[-1,i][:,:,:o1,:],bondO[:o1],2)
                tens2 = apply_bond(new_pepx[-1,i][:,:,o1:,:],bondO[o1:],2)
                new_site = tens1 + tens2
                new_bond = (bondO[:o1] + bondO[o1:])/2.
                new_site = apply_inverse_bond(new_site,new_bond,2)
                new_pepx[-1,i] = new_site.copy()
                new_lams[-1,i,2] = new_bond

    if obc[3]:   # right boundary
        for i in range(L1):
            r1 = pepx1[i,-1].shape[3]
            r2 = pepx2[i,-1].shape[3]

            if r1 == r2:
                bondR = get_bond(new_pepx,(i,-1),3)
                tens1 = apply_bond(new_pepx[i,-1][:,:,:,:r1],bondR[:r1],3)
                tens2 = apply_bond(new_pepx[i,-1][:,:,:,r1:],bondR[r1:],3)
                new_site = tens1 + tens2
                new_bond = (bondR[:r1] + bondR[r1:])/2.
                new_site = apply_inverse_bond(new_site,new_bond,3)
                new_pepx[i,-1] = new_site.copy()
                new_lams[i,-1,3] = new_bond

    return PEPX_GL(new_pepx, new_lams, pepx1.phys_bonds)


def axpby(alpha,pepx1,beta,pepx2):
    """
    return (alpha * mpx1) + (beta * mpx2)
    alpha, beta are scalar; mps1,mps2 are ndarrays of tensors (MPXs)
    """

    pepx_new = add(mul(alpha,pepx1),mul(beta,pepx))
    return pepx_new


def dot(pepx1, pepx2):
    """
    Computes PEPX1 * PEPX2   (PEPX1 above PEPX2)

    Parameters
    ----------
    mpx1: MPO or MPS
    mpx2: MPO or MPS

    Returns
    -------
    new_mpx : float or MPS or MPO
    """

    Ls = pepx1.shape
    assert pepx2.shape==Ls, '[dot]: sizes of pepx1 and pepx2 are not equal'
    new_pepx = np.empty(Ls, dtype=np.object)
    new_lams = np.empty(pepx1.lambdas.shape, dtype=np.object)

    # if np.all([ pepx1[i].ndim==3 and pepx2[i].ndim==3 for i in np.ndenumerate(pepx1) ]):
    #     return peps_dot(pepx1,pepx2)
    # else:
    for idx in np.ndindex(Ls):
        len_dp1 = len(pepx1.phys_bonds[idx])
        len_dp2 = len(pepx2.phys_bonds[idx])
        ax1 = [0,2,4,6] + range(8, 8+len_dp1)
        ax2 = [1,3,5,7] + range(8+len_dp1-1,8+len_dp1+len_dp2-1)
        ax2[-len_dp2] = ax1[-1]   # contract vertical bonds (mpx1 down with mpx2 up)
        new_site = np.einsum(pepx1[idx],ax1,pepx2[idx],ax2)
        new_pepx[idx] = tf.reshape(new_site,'ii,ii,ii,ii,...',group_ellipsis=False)

        i,j = idx
        for xx in range(new_lams.shape[2]):
            new_lams[i,j,xx] = np.outer(pepx1.lambdas[i,j,xx], pepx2.lambdas[i,j,xx]).reshape(-1)
            # print new_lams[i,j,xx].shape

    return PEPX_GL(new_pepx,new_lams) #,pepx1.phys_bonds)



def meas_obs(pepx_gl,pepo,ind0=(0,0),op_conn=None,pepx_type='peps',side='I',XMAX=100, envs=None, return_norm=False,
             contract_SL=False,scaleX=1):

    pepx = get_pepx(pepx_gl)
    return PEPX.meas_obs(pepx,pepo,ind0,op_conn,pepx_type,side,XMAX,envs,return_norm,contract_SL,scaleX=scaleX)


def meas_get_bounds(pepx_gl,bounds,op_shape,XMAX=100,contract_SL=False,scaleX=1):

    pepx = get_pepx(pepx_gl)
    return PEPX.meas_get_bounds(pepx,bounds,op_shape,XMAX,contract_SL,scaleX=scaleX)


def outer(pepx_u,pepx_d):

    Ls = pepx_u.shape

    out_gam = np.empty(Ls,dtype=np.object)
    out_lam = np.empty(Ls+(4,),dtype=np.object)

    for idx in np.ndindex(Ls):
        tens = np.einsum('lioru,LIORd->lLiIoOrRud',pepx_u[idx],pepx_d[idx])
        out_gam[idx] = tf.reshape(tens,'ii,ii,ii,ii,i,i')

        lams = np.empty(4,dtype=np.object)
        for xl in range(4):
            lams[xl] = np.outer(pepx_u.lambdas[idx][xl], pepx_d.lambdas[idx][xl]).reshape(-1)
        out_lam[idx] = lams

    return PEPX_GL(out_gam,out_lam)


############################################
######## QR factpring set-up with PEPX  #######
############################################ 


def get_io_axT(in_legs,out_legs,ndim,d_side='R'):
    ''' iso_legs:  str or list of 'l','u','d','r':  legs to be on the grouped with io axes and placed on right
        iso_side = right: (not iso),(iso + io)
        iso_side = left:  (isbo + io),(not iso)
        ndim = dimension of pepx tensor
    '''
 
    io_ax  = range(4,ndim)
   
    axL = []
    axM = []
    axR = []
    
    for ind in range(4):
        if   'lior'[ind] in in_legs:     axL.append(ind)   # left side of matrix
        elif 'lior'[ind] in out_legs:    axR.append(ind)   # right side of matrix
        else:                            axM.append(ind)

    if d_side == 'L':    axT = axL + io_ax + axM + axR
    elif d_side == 'R':  axT = axL + axM + io_ax + axR
    axT_inv = np.argsort(axT)   # back to ludr(io)

    return axT, axT_inv
 

def matricize(pepx_tens,iso_legs,iso_side):
    ''' peps tens is ndarray ludr(io)  
        iso_legs:  str or list of 'l','u','d','r':  legs to be on the grouped with io axes and placed on right
        iso_side = right: (not iso),(iso + io)
        iso_side = left:  (iso + io),(not iso)
    '''

    axT, axT_inv = get_io_axT(iso_legs,iso_side, peps_tens.ndim)
   
    if   iso_side in ['r','R','o','O','right']:
        axT, axT_inv = get_io_axT('', iso_legs, peps_tens.ndim)
        tens = peps_tens.transpose(axT)
        split_ind = -1*( len(iso_legs) + peps_tens.ndim-4 )
    elif iso_side in ['l','L','i','I','left']:
        axT, axT_inv = get_io_axT(iso_legs, '', peps_tens.ndim)
        tens = peps_tens.transpose(axT)
        split_ind = len(iso_legs) + peps_tens.ndim-4

    mat = tf.reshape(tens,split_ind)
    return mat, tens.shape, axT_inv


def unmatricize(peps_mat,tens_sh,axT_inv):
    '''  peps_mat is matrix (..., iso_legs + io)
         tens_sh:  shape of re-ordered tensor before reshaping
    '''
    tens = peps_mat.reshape(tens_sh).transpose(axT_inv)
    return tens



# def QR_factor(pepx_tens,iso):
# 
#     mat, mat_sh, axT_inv = matricize(pepx_tens,iso,'r',axT_inv=True)    
#     Q, R = np.linalg.qr(mat)
# 
#     Q = Q.reshape(mat_sh[0]+(-1,))
#     R = R.reshape((-1,)+mat_sh[1])
# 
#     return Q, R, axT_inv
#  
#    
# def LQ_factor(pepx_tens,iso):
# 
#     mat, mat_sh, axT_inv = matricize(pepx_tens,iso,'l',axT_inv=True)
#     Q, R = np.linalg.qr(mat.T)
# 
#     L = (R.T).reshape(mat_sh[0]+(-1,))
#     Q = (Q.T).reshape((-1,)+mat_sh[1])
# 
#     return L, Q, axT_inv


def QR_contract(Q,R,axT_inv):    
    tens = np.tensordot(Q,R,axes=(-1,0))
    return tens.transpose(axT_inv)
    

def LQ_contract(L,Q,axT_inv):
    tens = np.tensordot(L,Q,axes=(-1,0))
    return tens.transpose(axT_inv)


####################################################
#######  get chain of peps sites as list ###########
####################################################

##### lattice indexing fcts ######


def opposite_leg(leg):

    if   leg == 'l':  op_leg = 'r'
    elif leg == 'r':  op_leg = 'l'
    elif leg == 'o':  op_leg = 'i'
    elif leg == 'i':  op_leg = 'o'
    elif leg == 'u':  op_leg = 'd'
    elif leg == 'd':  op_leg = 'u'

    elif leg == 0:    op_leg = 3
    elif leg == 1:    op_leg = 2
    elif leg == 2:    op_leg = 1
    elif leg == 3:    op_leg = 0
    elif leg == 4:    op_leg = 5
    elif leg == 5:    op_leg = 4

    return op_leg


def opposite_legs(legs):

    if   isinstance(legs,str):     op_legs = ''
    elif isinstance(legs,list):    op_legs = []
    for leg in legs: 
        op_leg += opposite_leg(leg)

    return op_leg


def leg2ind(leg):
    if   leg == 'l':   return 0
    elif leg == 'i':   return 1
    elif leg == 'o':   return 2
    elif leg == 'r':   return 3 
    elif leg == 'u':   return 4
    elif leg == 'd':   return 5 
    else:              return leg   # stays int

def ind2leg(leg):
    if   leg == 0:     return 'l'
    elif leg == 1:     return 'i'
    elif leg == 2:     return 'o'
    elif leg == 3:     return 'r'
    elif leg == 4:     return 'u'
    elif leg == 5:     return 'd'
    else:              return leg   # stays str


def get_conn_inds(op_conn,ind0=(0,0)):
    ''' connectivity str -> [xs,ys] '''
    xs = (ind0[0],)
    ys = (ind0[1],)
    for x in op_conn:
        if   x in ['r','R',0]:
            xs = xs + (xs[-1],)
            ys = ys + (ys[-1]+1,)
        elif x in ['l','L',3]:
            xs = xs + (xs[-1],)
            ys = ys + (ys[-1]-1,)
        elif x in ['i','I',1]:
            xs = xs + (xs[-1]-1,)
            ys = ys + (ys[-1],)
        elif x in ['o','O',2]:
            xs = xs + (xs[-1]+1,)
            ys = ys + (ys[-1],)
    return [xs,ys]


def get_inds_conn(inds_list,wrap_inds=False,ind_type='str'):
    ''' get connection direction as progress in inds_list
        wrap_inds:  also return connection direction from last to first in list
        inds_list: [xs,ys]
    '''
    op_conns = ''

    for i in range(len(inds_list)-1):
        dx = inds_list[0][i+1]-inds_list[0][i]
        dy = inds_list[1][i+1]-inds_list[1][i]
        if   (dx,dy) == (0,1):   op_conns += 'r'
        elif (dx,dy) == (0,-1):  op_conns += 'l'
        elif (dx,dy) == (-1,0):  op_conns += 'i'
        elif (dx,dy) == (1,0):   op_conns += 'o'
        else:
            # print 'get_inds_conn: inds in inds_list are not connected'
            op_conns += ' '

    if ind_type == 'ints':   op_conns = [ leg2ind(leg) for leg in op_conns ]

    if wrap_inds:
        x0,y0 = (inds_list[0][0],  inds_list[1][0])
        xL,yL = (inds_list[0][-1], inds_list[1][-1])
        op_conns += get_inds_conn([[xL,x0],[yL,y0]])
        
    return op_conns


def get_conn_iolegs(connect_list, inds=False):
    
    in_legs  = []
    out_legs = []

    for ind in range(len(connect_list)+1):
        if ind == 0: in_legs.append( opposite_leg(connect_list[0]) )
        else:        in_legs.append( opposite_leg(out_legs[-1]) )  # opposite of previous out leg

        try:                out_legs.append( connect_list[ind] )
        except(IndexError): out_legs.append( opposite_leg(in_legs[-1]) )

    if inds:
        in_legs  = [leg2ind(leg) for leg in in_legs]
        out_legs = [leg2ind(leg) for leg in out_legs]

    return in_legs,out_legs


def get_conn_sharedlegs(connect_list, inds=False):

    if len(connect_list) <= 2:     return []
    # all cw 4 body terms
    elif connect_list in ['rol']:    return [((0,2),(3,1))]   # [((idx1, bond1), (idx2,bond2))]
    elif connect_list in ['oli']:    return [((0,0),(3,3))]   # [((idx1, bond1), (idx2,bond2))]
    elif connect_list in ['lir']:    return [((0,1),(3,2))]   # [((idx1, bond1), (idx2,bond2))]
    elif connect_list in ['iro']:    return [((0,3),(3,0))]   # [((idx1, bond1), (idx2,bond2))]
    else:
        xs,ys = get_conn_inds(connect_list)
        L = len(xs)
        shared_legs = []

        for ind in range(L):
           for i1 in range(ind+2,L): 
               op_conn = get_inds_conn([[xs[ind],xs[i1]],[ys[ind],ys[i1]]])
               if op_conn == ' ':  pass
               else:
                   idx1 = (ind,leg2ind(op_conn))
                   idx2 = (i1,leg2ind(opposite_leg(op_conn)))
                   shared_legs.append( (idx1,idx2) )
        # print 'shared legs', connect_list, shared_legs

        return shared_legs

        

def connect_pepx_list(pepx_list,connect_list,side='R'):
    ''' reorder peps tens axes such that they are x...dy ('R') or xd...y ('L')
        returns new_list, axT_invs
    '''

    L = len(pepx_list)

    new_list = []
    axTs = []
    for ind in range(0,L):
       
        try:   isoL = opposite_leg(isoR)
        except(NameError):    isoL = opposite_leg(connect_list[0])
       
        try:   isoR = connect_list[ind]
        except(IndexError):   isoR = opposite_leg(isoL)

        axT,axT_inv = get_io_axT(isoL,isoR,pepx_list[ind].ndim,d_side=side)
        tens = pepx_list[ind].transpose(axT)
        new_list.append(tens)

        axTs.append(axT_inv)

    return new_list, axTs


def transpose_pepx_list(pepx_list,axTs):
    ''' reorders list of pepx tens according to axT_inv '''

    new_list = []
    for ind in range(len(pepx_list)):
        new_list.append(pepx_list[ind].transpose(axTs[ind]))

    return new_list


# def conj_transpose_pepx_list(pepx_list,axTs):
#     ''' reorders list of pepx tens according to axT_inv '''
# 
#     new_list = []
#     for ind in range(len(pepx_list)):
#         try: 
#             tens = np.conj(np.transpose(pepx_list[ind],[0,1,2,3,5,4]))
#         except(ValueError):   # tens is not 6 dimensional
#             tens = np.conj(pepx_list[ind])
# 
#         new_list.append(tens)
# 
#     return new_list



def get_bond(pepx,idx,direction):
    ''' idx:  site index, direction is string or int denoting leg being updated '''

    i,j= idx

    if isinstance(direction,str):
        direction = leg2ind(direction)

    bond = pepx.lambdas[i,j,direction]
    return bond


def set_bond(pepx,idx,direction,new_bond):

    i,j = idx
    L1,L2 = pepx.shape

    try:
        if direction in ['r','R',3]:
            pepx.lambdas[i,j,  3] = new_bond[:]
            if j < L2-1:  pepx.lambdas[i,j+1,0] = pepx.lambdas[i,j,3]
        elif direction in ['o','O',2]:
            pepx.lambdas[i,  j,2] = new_bond[:]
            if i < L1-1:  pepx.lambdas[i+1,j,1] = pepx.lambdas[i,j,2]
        elif direction in ['i','I',1]:
            pepx.lambdas[i,  j,1] = new_bond[:]
            if i > 0:     pepx.lambdas[i-1,j,2] = pepx.lambdas[i,j,1]
        elif direction in ['l','L',0]:
            pepx.lambdas[i,j  ,0] = new_bond[:]
            if j > 0:     pepx.lambdas[i,j-1,3] = pepx.lambdas[i,j,0]
    except(IndexError):
        pass
        # # i think these (at i,j) should already have been set?
        # if direction in ['r','R',3]:
        #     pepx.lambdas[i,j,  3] = new_bond[:]
        # elif direction in ['o','O',2]:
        #     pepx.lambdas[i,  j,2] = new_bond[:]
        # elif direction in ['i','I',1]:
        #     pepx.lambdas[i,  j,1] = new_bond[:]
        # elif direction in ['l','L',0]:
        #     pepx.lambdas[i,j  ,0] = new_bond[:]

    # print 'set bonds', [m.shape for m in pepx.lambdas[i,j]]

    return pepx


def check_bonds(pepx):

    L1, L2 = pepx.shape

    for (i,j) in np.ndindex(L1,L2):
        
         if i > 0:    assert( np.all(pepx.lambdas[i,j,1] == pepx.lambdas[i-1,j,2]) ),'(%d,%d) io bond error'%(i,j)
         if i < L1-1: assert( np.all(pepx.lambdas[i,j,2] == pepx.lambdas[i+1,j,1]) ),'(%d,%d) io bond error'%(i,j)
         if j > 0:    assert( np.all(pepx.lambdas[i,j,0] == pepx.lambdas[i,j-1,3]) ),'(%d,%d) io bond error'%(i,j)
         if j < L2-1: assert( np.all(pepx.lambdas[i,j,3] == pepx.lambdas[i,j+1,0]) ),'(%d,%d) io bond error'%(i,j)

    return


def apply_bond(pepx_tens,bond,leg_ax):
    ''' apply diagonal vec to leg of peps tens'''

    if   leg_ax in ['l','L',0]:
        new_tens = np.einsum('lior...,lL->Lior...',pepx_tens,np.diag(bond))
    elif leg_ax in ['i','I',1]:
        new_tens = np.einsum('lior...,iI->lIor...',pepx_tens,np.diag(bond))
    elif leg_ax in ['o','O',2]:
        new_tens = np.einsum('lior...,oO->liOr...',pepx_tens,np.diag(bond))
    elif leg_ax in ['r','R',3]:
        new_tens = np.einsum('lior...,rR->lioR...',pepx_tens,np.diag(bond))

    return new_tens


def apply_inverse_bond(pepx_tens,bond,leg_ax):
    ''' apply diagonal vec to leg of peps tens'''

    if   leg_ax in ['l','L',0]:
        new_tens = np.einsum('lior...,lL->Lior...',pepx_tens,np.diag(1./bond))
    elif leg_ax in ['i','I',1]:
        new_tens = np.einsum('lior...,iI->lIor...',pepx_tens,np.diag(1./bond))
    elif leg_ax in ['o','O',2]:
        new_tens = np.einsum('lior...,oO->liOr...',pepx_tens,np.diag(1./bond))
    elif leg_ax in ['r','R',3]:
        new_tens = np.einsum('lior...,rR->lioR...',pepx_tens,np.diag(1./bond))

    return new_tens


def apply_lam_to_site(site_tens,bonds,no_lam=[],op='default'):

    new_tens = site_tens.copy()

    # print 'apply lam', new_tens.shape, [m.shape for m in bonds]

    if op == 'sqrt':
        no_axes = [leg2ind(leg) for leg in no_lam]
        for leg_ax in range(4):
            if leg_ax not in no_axes:
                new_tens = apply_bond(new_tens,np.sqrt(bonds[leg_ax]),leg_ax)

    elif op == 'inv':
        no_axes = [leg2ind(leg) for leg in no_lam]
        for leg_ax in range(4):
            if leg_ax not in no_axes:
                new_tens = apply_bond(new_tens,1./bonds[leg_ax],leg_ax)

    elif op == 'sqrt_inv':
        no_axes = [leg2ind(leg) for leg in no_lam]
        for leg_ax in range(4):
            if leg_ax not in no_axes:
                new_tens = apply_bond(new_tens,1./np.sqrt(bonds[leg_ax]),leg_ax)

    else:
        no_axes = [leg2ind(leg) for leg in no_lam]
        for leg_ax in range(4):
            if leg_ax not in no_axes:
                new_tens = apply_bond(new_tens,bonds[leg_ax],leg_ax)

    return new_tens


def get_site(pepx,idx,no_lam=[],op='default'):

    return apply_lam_to_site(pepx[idx],pepx.lambdas[idx],no_lam,op)


def get_sites(pepx,ind0,connect_list):

    xs,ys = get_conn_inds(connect_list,ind0)
    in_legs, out_legs = get_conn_iolegs(connect_list,inds=True)
    # shared_legs = get_conn_sharedlegs(connect_list)

    gammas  = []
    lambdas = []

    x,y = xs[0], ys[0]
    lambdas.append( pepx.lambdas[x,y,in_legs[0]] )

    for ind in range(len(xs)):
        x,y = xs[ind], ys[ind]

        gammas.append( get_site(pepx,(x,y),no_lam=[in_legs[ind],out_legs[ind]]) )
        lambdas.append( pepx.lambdas[x,y,out_legs[ind]] )

    # for ind in range(len(shared_legs)):
    #     s1,s2 = shared_legs[ind]
    #     idx1,leg1 = s1
    #     idx2,leg2 = s2
    #     x1,y1 = xs[idx1],ys[idx1]
    #     x2,y2 = xs[idx2],ys[idx2]
    #     lam1 = pepx.lambdas[x1,y1,leg1]
    #     lam2 = pepx.lambdas[x2,y2,leg2]
    #     assert(np.all(lam1==lam2))

    #     g1 = apply_bond( gammas[idx1], 1./np.sqrt(lam1), leg1)
    #     g2 = apply_bond( gammas[idx2], 1./np.sqrt(lam2), leg2)

    #     gammas[idx1] = g1
    #     gammas[idx2] = g2

    gammas, axT_invs = connect_pepx_list(gammas,connect_list,'R')   # gammas now in x ... x d x out

    return gammas, lambdas, axT_invs


def remove_lam_from_site(site_tens,bonds,no_lam=[]):

    return apply_lam_to_site(site_tens,bonds,no_lam,op='inv')

    # new_site = site_tens.copy()
    # 
    # # remoorioriginal bonds from new_site except for those specified by no_lam
    # no_axes = [leg2ind(leg) for leg in no_lam]
    # for leg_ax in range(4):
    #     if leg_ax not in no_axes:
    #         new_site = apply_bond(new_site,1./bonds[leg_ax],leg_ax)

    # return new_site
           

def set_sites(pepx,ind0,connect_list,g_list,l_list,axT_invs):
 
    pepx_ = pepx.copy()

    xs,ys = get_conn_inds(connect_list,ind0)
    in_legs, out_legs = get_conn_iolegs(connect_list,inds=True)
    shared_legs = get_conn_sharedlegs(connect_list)    

    new_gs = transpose_pepx_list(g_list,axT_invs)

    # pepx_.lambdas[x,y,in_legs[0]] = l_list[0]   # shouldn't have changed

    for ind in range(len(g_list)):
        x,y = xs[ind],ys[ind]
        in_leg = in_legs[ind]
        out_leg = out_legs[ind]

        # remove lams not along the pepx_list path
        stripped_tens = remove_lam_from_site(new_gs[ind],pepx.lambdas[x,y],no_lam=[in_leg,out_leg])

        # print 'set sites', x,y, pepx.lambdas[x,y], np.linalg.norm(stripped_tens), np.linalg.norm(new_gs[ind])
        # print 'set sites', l_list

        # print [m.shape for m in pepx.lambdas[x,y]]
        # print 'set sites', ind0, x,y, stripped_tens.shape, out_leg, len(l_list[ind+1])

        pepx_[x,y] = stripped_tens
        pepx_ = set_bond(pepx_,(x,y),out_leg,l_list[ind+1])


    # for ind in range(len(g_list)):
    #     x,y = xs[ind], ys[ind]
    #     print 'set to', connect_list, np.linalg.norm(pepx_[x,y]), pepx_.lambdas[x,y], l_list[ind+1]
        

    for ind in range(len(shared_legs)):
        s1,s2 = shared_legs[ind]
        idx1,leg1 = s1
        idx2,leg2 = s2
        x1,y1 = xs[idx1],ys[idx1]
        x2,y2 = xs[idx2],ys[idx2]
        lam1 = pepx.lambdas[x1,y1,leg1]
        lam2 = pepx.lambdas[x2,y2,leg2]
        assert(np.all(lam1==lam2))

        g1 = apply_bond( pepx_[x1,y1], np.sqrt(lam1), leg1 )
        g2 = apply_bond( pepx_[x2,y2], np.sqrt(lam2), leg2 )

        pepx_[x1,y1] = g1
        pepx_[x2,y2] = g2
 
    return pepx_
    

def GL_to_pepx_list(g_list,l_list,canon='L'):

    pepx_list = []

    if canon in ['L',0]:      # left canonical
        for ind in range(len(g_list)):
            ptens = g_list[ind]
            ptens = tf.dMult('DM',l_list[ind],ptens)
            pepx_list.append( ptens )
       
        # include end bond on last site
        ptens = tf.dMult('MD',ptens,l_list[ind+1])
        pepx_list[-1] = ptens

    elif canon in ['R',1]:    # right canonical
        for ind in range(len(g_list)):
            ptens = g_list[ind]
            ptens = tf.dMult('MD',ptens,l_list[ind+1])
            pepx_list.append( ptens )

        # include end bond on first site
        ptens = pepx_list[0]
        ptens = tf.dMult('DM',l_list[0],ptens)
        pepx_list[0] = ptens

    elif canon in ['M',2]:     # bonds are split between sites (regularized)
        raise(NotImplementedError)

    return pepx_list


def pepx_to_GL_list(pepx_list,lamL=None,lamR=None,direction=0,DMAX=-1,num_io=1,canonicalize=True,normalize=True,
                    tol=1.0e-12):

    L = len(pepx_list)

    gamma_new = []
    lambda_new = []
    errs = []

    new_list = pepx_list[:]

    if lamL is None:   lamL = np.ones(pepx_list[0].shape[0])
    if lamR is None:   lamR = np.ones(pepx_list[-1].shape[-1])

    # if normalize:
    #     norm = PEPX.norm_pepx_list(new_list)
    #     new_list = PEPX.mul_pepx_list(new_list,1./norm)
    #     # print 'new norm', norm, PEPX.norm_pepx_list(new_list)

    norm = PEPX.norm_pepx_list(new_list)
    new_list = PEPX.mul_pepx_list(new_list,1./norm)

    if direction == 0:
     
        if canonicalize:
            new_list, temp = PEPX.compress_peps_list(new_list,None,DMAX=-1,num_io=num_io,direction=1,regularize=False)
            # print 'pepx to GL canon err', temp
            # print 'pepx to gl canon?', PEPX.check_canon_pepx_list(new_list[1:],canon='R')

        for ind in range(0,L-1):
            block = np.tensordot(new_list[ind],new_list[ind+1],axes=(-1,0))
            u,s,vt,dwt = tf.svd(block,new_list[ind].ndim-1,DMAX,tol=tol)
            # u,s,vt,dwt = tf.svd(block,block.ndim/2,DMAX)
            new_list[ind+1] = tf.dMult('DM',s,vt)

            # print 'pepx to GL list block norm', np.linalg.norm(block), np.linalg.norm(s)



            try:
                gamma_new.append( tf.dMult('DM', 1./lambda_new[ind-1],u) )
            except(IndexError):
                gamma_new.append( tf.dMult('DM', 1./lamL, u) )    # 'left edge'

            lambda_new.append(s)
            errs.append(dwt)

        gamma_new.append( tf.dMult('MD', vt, 1./lamR) )
        # u_,s_,vt_,dwt = tf.svd(tf.dMult('DM',s,vt),vt.ndim-1,DMAX)
        # print 'pepx to GL xx', vt.shape, s_, lamR, vt_

    elif direction == 1:

        if canonicalize:
            new_list, temp = PEPX.compress_peps_list(new_list,None,DMAX=-1,num_io=num_io,direction=0,regularize=False)
            # print 'pepx to gl canon?', PEPX.check_canon_pepx_list(new_list[:-1],canon='L')

        for ind in range(L-1,0,-1):
            block = np.tensordot(new_list[ind-1],new_list[ind],axes=(-1,0))
            u,s,vt,dwt = tf.svd(block,new_list[ind-1].ndim-1,DMAX,tol=tol)
            # u,s,vt,dwt = tf.svd(block,block.ndim/2,DMAX)
            new_list[ind-1] = tf.dMult('MD',u,s)
            
            try:
                gamma_new.append( tf.dMult('MD', vt, 1./lambda_new[L-ind-2]) )
            except(IndexError):
                gamma_new.append( tf.dMult('MD', vt, 1./lamR) )

            lambda_new.append(s)
            errs.append(dwt)

        gamma_new.append( tf.dMult('DM', 1./lamL, u) )

        gamma_new = gamma_new[::-1]
        lambda_new = lambda_new[::-1]
        errs = errs[::-1]

    lambda_new = [lamL] + lambda_new + [lamR]

    if not normalize:
        gamma_new, lambda_new = mul_GL_list(gamma_new,lambda_new,norm)
        temp_list = GL_to_pepx_list(gamma_new,lambda_new)

    return gamma_new,lambda_new, errs



def get_pepx(pepx_gl,canon='M',outer_lam='full'):
    ''' return PEPX object; bonds [except edge bonds] are split via sqrt'''

    L1,L2 = pepx_gl.shape
    pepx_ = np.empty((L1,L2),dtype=np.object)

    if canon in ['L','LI']:   # left and in canonical
        for idx in np.ndindex((L1,L2)):
            bonds = pepx_gl.lambdas[idx]
            pepx_[idx] = apply_lam_to_site(pepx_gl[idx], bonds, no_lam=[2,3])

            if idx[0] == L1-1:    # bottom row
                pepx_[idx] = apply_bond(pepx_[idx],bonds[2],2)
            if idx[1] == L2-1:    # right col
                pepx_[idx] = apply_bond(pepx_[idx],bonds[3],3)

    elif canon == 'LO':       # left and down canonical
        for idx in np.ndindex((L1,L2)):
            bonds = pepx_gl.lambdas[idx]
            pepx_[idx] = apply_lam_to_site(pepx_gl[idx], bonds, no_lam=[1,3])

            if idx[0] == 0:       # top row
                pepx_[idx] = apply_bond(pepx_[idx],bonds[1],1)
            if idx[1] == L2-1:    # right col
                pepx_[idx] = apply_bond(pepx_[idx],bonds[3],3)
        
    elif canon in ['R','RO']: # right and out canonical
        for idx in np.ndindex((L1,L2)):
            bonds = pepx_gl.lambdas[idx]
            pepx_[idx] = apply_lam_to_site(pepx_gl[idx], bonds, no_lam=[0,1])

            if idx[0] == 0  :     # top row
                pepx_[idx] = apply_bond(pepx_[idx],bonds[1],1)
            if idx[1] == 0:       # left col
                pepx_[idx] = apply_bond(pepx_[idx],bonds[0],0)

    elif canon == 'RI':       # right and in canonical
        for idx in np.ndindex((L1,L2)):
            bonds = pepx_gl.lambdas[idx]
            pepx_[idx] = apply_lam_to_site(pepx_gl[idx], bonds, no_lam=[0,2])

            if idx[0] == L1-1:    # bottom row
                pepx_[idx] = apply_bond(pepx_[idx],bonds[2],2)
            if idx[1] == 0:       # left col
                pepx_[idx] = apply_bond(pepx_[idx],bonds[0],0)
        
    else:

        for idx in np.ndindex((L1,L2)):
            bonds = pepx_gl.lambdas[idx]
            pepx_[idx] = apply_lam_to_site(pepx_gl[idx], bonds, op='sqrt')

            if outer_lam == 'full':
                if idx[0] == 0:      # top row
                    pepx_[idx] = apply_bond(pepx_[idx],np.sqrt(bonds[1]),1)
                if idx[0] == L1-1:   # bottom row
                    pepx_[idx] = apply_bond(pepx_[idx],np.sqrt(bonds[2]),2)
                if idx[1] == 0:      # left col
                    pepx_[idx] = apply_bond(pepx_[idx],np.sqrt(bonds[0]),0)
                if idx[1] == L2-1:   # right col
                    pepx_[idx] = apply_bond(pepx_[idx],np.sqrt(bonds[3]),3)

            elif outer_lam=='sqrt':
                pepx_[idx] = pepx_[idx]

            else:
                if idx[0] == 0:      # top row
                    pepx_[idx] = apply_bond(pepx_[idx],1./np.sqrt(bonds[1]),1)
                if idx[0] == L1-1:   # bottom row
                    pepx_[idx] = apply_bond(pepx_[idx],1./np.sqrt(bonds[2]),2)
                if idx[1] == 0:      # left col
                    pepx_[idx] = apply_bond(pepx_[idx],1./np.sqrt(bonds[0]),0)
                if idx[1] == L2-1:   # right col
                    pepx_[idx] = apply_bond(pepx_[idx],1./np.sqrt(bonds[3]),3)


    return PEPX.PEPX(pepx_,pepx_gl.phys_bonds)


def mul_GL_list(gammas,lambdas,const):
    
    L = len(gammas)

    new_gammas  = [g.copy() for g in gammas]
    new_lambdas = [l.copy() for l in lambdas]

    for x in range(L):
        new_gammas[x] = new_gammas[x]*(np.abs(const)**(1./L))
 
    ### if scale lambdas, can lead to unphysically large lambdas ###
    # for x in range(1,L):
    #     new_lambdas[x] = lambdas[x]*(np.abs(const)**(1./(L-1)))
    

    dtype = np.result_type(const,gammas[0])
    if dtype == int or dtype == float:
        phase = np.sign(const)
    elif dtype == complex:
        phase = np.exp(1j*np.angle(const))
    new_gammas[0] = new_gammas[0]*phase

    return new_gammas, new_lambdas


def regularize_GL_list(gammas,lambdas):
    ''' ignore lambdas[0], lambdas[-1].  rescale gammas s.t. sum lambdas**2 = 1'''

    new_gammas  = gammas[:]   #[g.copy() for g in gammas]
    new_lambdas = lambdas[:]  #[l.copy() for l in lambdas]

    for ind in range(1,len(lambdas)-1):
        norm = np.linalg.norm(lambdas[ind])
        # new_gammas[ind-1] = (norm**0.25)*new_gammas[ind-1]
        # new_gammas[ind]   = (norm**0.25)*new_gammas[ind]
        # new_lambdas[ind]  = new_lambdas[ind]/np.sqrt(norm)
        new_gammas[ind-1] = (norm**0.5)*new_gammas[ind-1]
        new_gammas[ind]   = (norm**0.5)*new_gammas[ind]
        new_lambdas[ind]  = new_lambdas[ind]/norm

    return new_gammas, new_lambdas


def full_to_qr_GL_list(gam_list,lam_list,num_io=1):
    ''' take qr/lq of first/last elements in GL_list '''

    tens0 = tf.dMult('DM',lam_list[0], gam_list[0])
    tensL = tf.dMult('MD',gam_list[-1],lam_list[-1])

    q0,r0 = tf.qr(tens0,3)  #4-num_io)
    qL,rL = tf.qr( np.swapaxes( tensL, 0, -1 ),3) # 4-num_io )   # i...do -> o...di -> o...q/qdi
    # lL,qL = tf.lq(np.moveaxis(tensL,[-2-m for m in range(num_io)[::-1]],[1+m for m in range(num_io)]),1+num_io)
        # move phys bonds to right after input
        # should probably double check tf.lq method

    # print 'full to qr', tens0.shape, tensL.shape
    # print 'full to qr', q0.shape, qL.shape

    gam_list_qr = gam_list[:]
    lam_list_qr = lam_list[:]
    gam_list_qr[0]  = r0  
    gam_list_qr[-1] = np.swapaxes( rL, 0, -1 )
    lam_list_qr[0]  = np.ones(r0.shape[0])
    lam_list_qr[-1] = np.ones(rL.shape[0])

    return gam_list_qr, lam_list_qr, q0, qL   # q leg is always at end


def qr_to_full_GL_list(gam_list_qr,lam_list_qr,q0,qL,lam0,lamL):

    tens0 = np.tensordot(q0, gam_list_qr[0], axes=(-1,0))
    tensL = np.tensordot(qL, np.swapaxes(gam_list_qr[-1],0,-1), axes=(-1,0))  # q leg is at end
    tensL = np.swapaxes( tensL, -1, 0 )

    gamma0 = tf.dMult('DM',1./lam0,tens0)
    gammaL = tf.dMult('MD',tensL,1./lamL)

    gam_list = [gamma0] + gam_list_qr[1:-1] + [gammaL]
    lam_list = [lam0] + lam_list_qr[1:-1] + [lamL]

    return gam_list, lam_list
    

def compress_GL_list(gammas,lambdas,DMAX=-1,num_io=1,direction=0,normalize=True,tol=1.0e-12):

    if direction==0:  canon = 'R'
    else:             canon = 'L'

    pepx_list = GL_to_pepx_list(gammas,lambdas,canon=canon)
    norm = PEPX.norm_pepx_list(pepx_list)
    pepx_list = PEPX.mul_pepx_list(pepx_list,1./norm)
    new_gams, new_lams, errs = pepx_to_GL_list(pepx_list,lamL=lambdas[0],lamR=lambdas[-1],direction=direction,
                                               DMAX=DMAX,num_io=num_io,canonicalize=True,normalize=False,tol=tol)

    if not normalize:
        new_gams, new_lams = mul_GL_list(new_gams, new_lams, norm)

    return new_gams, new_lams, errs


def canonicalize_GL_list(gammas,lambdas,direction=0,num_io=1,normalize=True,tol=1.0e-12):
 
    new_gams, new_lams, errs = compress_GL_list(gammas,lambdas,DMAX=-1,num_io=num_io,
                                                direction=direction,normalize=normalize,tol=tol)

    return new_gams, new_lams
   
    # if direction==0:  canon = 'R'
    # else:             canon = 'L'

    # pepx_list = GL_to_pepx_list(gammas,lambdas,canon=canon)
    # norm = PEPX.norm_pepx_list(pepx_list)
    # pepx_list = PEPX.mul_pepx_list(pepx_list,1./norm)
    # new_gams, new_lams, errs = pepx_to_GL_list(pepx_list,lamL=lambdas[0],lamR=lambdas[-1],direction=direction,
    #                                            DMAX=-1,canonicalize=True,normalize=False)
    # if not normalize:
    #     new_gams, new_lams = mul_GL_list(new_gams, new_lams, norm)

    # return new_gams, new_lams


def check_GL_canonical(gammas,lambdas,check='LR',kill_calc=False):   # 'llld(d)r'

    # for lam in lambdas:  print 'sum lam', lam, np.linalg.norm(lam)
    stop_calc = False

    if 'L' in check:
        left_canon = []
        for ind in range(len(gammas)):
            left_canon += [tf.dMult('DM',lambdas[ind],gammas[ind])]

        xx = 0
        for LL in left_canon:
            # try:                  LL_left = np.einsum('ij,j...->i...',LL_, LL)
            # except(NameError):    LL_left = LL   # leftmost
            LL_left = LL
            LL_ = np.tensordot( np.conj(LL), LL_left, axes=(range(LL.ndim-1),range(LL.ndim-1)) )
            if not np.allclose(LL_, np.eye(LL_.shape[0])):
                print 'G/L not left canonical', xx
                # print LL_
                # print np.einsum('ii->', tf.dMult('MD',tf.dMult('DM',np.conj(lambdas[xx+1]),LL_),lambdas[xx+1]))
                # print np.linalg.norm(lambdas[xx+1])
                stop_calc = True
                # print np.abs( LL_ - np.eye(LL_.shape[0]) )
            xx += 1

        temp = np.einsum('ii->', tf.dMult('MD',tf.dMult('DM',np.conj(lambdas[-1]),LL_),lambdas[-1]))
        normL = np.sqrt(temp)
    
    if 'R' in check:
        right_canon = []
        for ind in range(len(gammas)):
            right_canon += [tf.dMult('MD',gammas[ind],lambdas[ind+1])]

        xx = len(gammas)
        for RR in right_canon[::-1]:
            RR_ = np.tensordot( np.conj(RR), RR, axes=(range(1,RR.ndim),range(1,RR.ndim)) )
            # print 'RR',RR_
            if not np.allclose(RR_, np.eye(RR_.shape[0])):
                print 'G/L not right canonical',xx-1
                # print RR_
                stop_calc = True
                # print np.abs( RR_ - np.eye(RR_.shape[0]))
            xx -= 1

        temp = np.einsum('ii->', tf.dMult('MD',tf.dMult('DM',np.conj(lambdas[0]),RR_),lambdas[0]))
        normR = np.sqrt(temp)


    if stop_calc:
        try:           print 'norm L', normL
        except:        pass
        try:           print 'norm R', normR
        except:        pass
        if kill_calc:  raise RuntimeError('PEPX_GL: not canonicalized')


# def regularize_loop(pepx, ind0, conn_list=None, indc_list=None):
# 
#     i,j = ind0
# 
#     new_pepx = pepx.copy()
# 
#     for idx in np.ndindex(2,2):
#         i0,j0 = idx
#         # print 'regularize 1',idx, new_pepx.lambdas[i+i0,j+j0]
# 
# 
#     if conn_list is None:
#         conn_list = ['rol']   # ['oli','lir','iro','rol']
#         indc_list = [(0,0)]   # [(0,1),(1,1),(1,0),(0,0)]
# 
#     for ind in range(len(conn_list)):
# 
#         new_ij = (i+indc_list[ind][0],j+indc_list[ind][1])
# 
#         sub_gammas, sub_lambdas, axTs = get_sites(new_pepx,new_ij,conn_list[ind])
#         # print 'orig sub lambdas', sub_lambdas
# 
#         sh_1 = [m.shape for m in sub_lambdas]
#    
#         temp_pepx = GL_to_pepx_list(sub_gammas,sub_lambdas,canon='L')
# 
#         nd = temp_pepx[0].ndim
#         pepx_block = temp_pepx[0]
#         for xx in range(1,len(temp_pepx)):
#             pepx_block = np.tensordot(pepx_block,temp_pepx[xx],axes=(-1,0))
# 
# 
#         lamL,lamR = sub_lambdas[0], sub_lambdas[-1]
#         sub_gammas, sub_lambdas, errs = tf.decompose_block_GL(pepx_block,len(temp_pepx),0,
#                                                               lamL=lamL,lamR=lamR,svd_str=nd-1)
#         # print 'new sub lambdas', sub_lambdas
#         # print 'regularize loop', errs
# 
#         sh_2 = [m.shape for m in sub_lambdas]
# 
#         if np.any( sh_1 != sh_2 ):
#             print 'lambda bond dimension changed !!!!!!'
#             print sh_1, sh_2
#             # exit()
# 
#         new_pepx = set_sites(new_pepx,new_ij,conn_list[ind],sub_gammas,sub_lambdas,axTs)
# 
#     # for idx in np.ndindex(2,2):
#         # i0,j0 = idx
#         # print 'regularize 2', idx, new_pepx.lambdas[i+i0,j+j0]
# 
# 
#     return new_pepx


def regularize_loop(pepx, ind0, conn_list=None, indc_list=None):

    i,j = ind0

    new_pepx = pepx.copy()

    for idx in np.ndindex(2,2):
        i0,j0 = idx
        # print 'regularize 1',idx, new_pepx.lambdas[i+i0,j+j0]


    if conn_list is None:
        conn_list = ['ori']   # ['oli','lir','iro','rol']
        indc_list = [(0,0)]   # [(0,1),(1,1),(1,0),(0,0)]

    for ind in range(len(conn_list)):

        new_ij = (i+indc_list[ind][0],j+indc_list[ind][1])

        sub_gammas, sub_lambdas, axTs = get_sites(new_pepx,new_ij,conn_list[ind])
        # print 'orig sub lambdas', sub_lambdas

        sh_1 = [m.shape for m in sub_lambdas]
   
        temp_pepx = GL_to_pepx_list(sub_gammas,sub_lambdas,canon='L')
        lamL,lamR = sub_lambdas[0], sub_lambdas[-1]
        # print 'reg loop', [m.shape for m in sub_gammas], [m.shape for m in sub_lambdas]
        sub_gammas, sub_lambdas, errs = pepx_to_GL_list(temp_pepx,lamL,lamR,direction=1,canonicalize=False)
        # print 'new sub lambdas', sub_lambdas
        # print 'regularize loop', errs

        sh_2 = [m.shape for m in sub_lambdas]

        if np.any( sh_1 != sh_2 ):
            print 'lambda bond dimension changed !!!!!!'
            print sh_1, sh_2
            # exit()

        new_pepx = set_sites(new_pepx,new_ij,conn_list[ind],sub_gammas,sub_lambdas,axTs)

    for idx in np.ndindex(2,2):
        i0,j0 = idx
        # print 'regularize 2', idx, new_pepx.lambdas[i+i0,j+j0]


    return new_pepx


###########################################################
#### apply operators (trotter steps) to PEPX or PEPO ######
###########################################################

def mpo_update(gamma_list, lambda_list, mpo1, mpo2=None, DMAX=100, direction=0, num_io=1, normalize=False,
               chk_canon=False):
    ''' algorithm:  mpo1 is already decomposed... so just need to do contractions + compression
                    but maybe mpo representation of exp() is not great? (compression of mpo's not so well behaved?)
        contraction + compression:  qr + svd compression
        connect_list:  string of 'l','u','d','r' denoting path of mpo and pepx_list
    '''

    L = len(gamma_list)
    # print 'mpo update len', L, connect_list

    new_list = []

    lamL = lambda_list[0]
    lamR = lambda_list[-1]


    for ind in range(L):

        tens = gamma_list[ind]
        tens_ = tf.dMult('DM',lambda_list[ind], tens)

        # this way we can more easily integrate applying mpo to (Q)R gammas
        if num_io == 1:
            try:
                tens_ = np.einsum('LudR,l...dx->Ll...uRx',mpo1[ind],tens_)
                tens_ = tf.reshape(tens_,'ii,...,ii')
            except(TypeError):   # mpo1 is None
                pass
        elif num_io == 2:
            try:
                tens_ = np.einsum('LudR,l...dDx->Ll...uDRx',mpo1[ind],tens_)
                tens_ = tf.reshape(tens_,'ii,...,ii')
            except(TypeError):   # mpo1 is None
                pass
            try:
                tens_ = np.einsum('LUDR,l...dUx->lL...dDxR',mpo2[ind],tens_)
                tens_ = tf.reshape(tens_,'ii,...,ii')
            except(TypeError):   # mpo2 is None
                pass
        else:
            raise(NotImplementedError)

        # try:
        #     tens_ = np.einsum('LudR,labd...x->Llabu...Rx',mpo1[ind],tens_) 
        #     tens_ = tf.reshape(tens_,'ii,...,ii')
        # except(TypeError):   # mpo1 is None
        #     pass
        # try: 
        #     tens_ = np.einsum('LdDR,l...dx->lL...DxR',mpo2[ind],tens_)
        #     tens_ = tf.reshape(tens_,'ii,...,ii')
        # except(TypeError):   # mpo2 is none
        #     pass

        new_list.append(tens_)

    # last site
    new_list[-1] = tf.dMult('MD',new_list[-1],lambda_list[-1])

    gamma_new, lambda_new, errs = pepx_to_GL_list(new_list,lamL,lamR,DMAX=DMAX,direction=0,num_io=num_io,
                                                  canonicalize=True,normalize=normalize)

    # print 'mpo update errs', errs, DMAX, len(lambda_new[1])

    if chk_canon:
        print 'mpo update check canon'
        check_GL_canonical(gamma_new,lambda_new)    # looks good...
        print 'done mpo update check canon'


    if np.any( [np.linalg.norm(m) > 12 for m in lambda_new] ):
        print 'large lambda'
        # exit()

    if np.all( [np.linalg.norm(m) < 1.0e-8 for m in lambda_new] ):
        print 'small lambda'
        # exit()


    return gamma_new, lambda_new, errs



# def update_loop(sub_pepx, mpo1=None, ind0=(0,0), connect_list=None, DMAX=10, mpo2=None, normalize=True):
# 
#     assert(sub_pepx.shape == (2,2)), 'sub pepx needs to be 2x2 loop'
# 
#     new_pepx_gl = sub_pepx.copy()
#     pepx_ = get_pepx(new_pepx_gl)
# 
#     ### application of MPOs ####
#     if mpo1 is not None: 
#        pepx_list, axTs = PEPX.get_sites(pepx_,ind0,connect_list)
#        new_list, errs = PEPX.mpo_update(pepx_list, None, mpo1, mpo2=mpo2, DMAX=-1, direction=2, regularize=False)
#                         # direction != 0, 1 --> no compression (just application of mpo)
#        new_pepx = PEPX.set_sites(pepx_,new_list,ind0,connect_list,axTs)
#     else:
#        new_pepx = pepx_
# 
#     ### contract loop ###
#     if new_pepx[0,0].ndim == 5:
#         loop = np.einsum('liord,rjpse->liodjpse',new_pepx[0,0],new_pepx[0,1])
#         loop = np.einsum('liodjpse,LoORD->lidjpseLORD',loop,new_pepx[1,0])
#         loop = np.einsum('lidjpseLORD,RpPSE->lidjseLODPSE',loop,new_pepx[1,1])
# 
#         if normalize:
#             norm = np.linalg.norm(loop)
#             loop = 1./norm * loop
# 
#         ### top and bottom rows ###
# 
#         u,s,vt,dwt = tf.svd(loop,loop.ndim/2,2*DMAX)    # (0,0)+(0,1) , (1,0)+(1,1)
#   
#         two_site1 = tf.dMult('MD',u,s)
#         two_site2 = tf.dMult('DM',s,vt)
# 
#         b1dim = len(s)
#         b2dim = int(np.sqrt(b1dim))
# 
#         iden = np.eye(b1dim)[:,:b2dim**2]
#         isometry = iden.reshape(b1dim,b2dim,b2dim)
# 
#         two_site1 = np.einsum('lidjseX,Xop->liodjpse',two_site1,isometry)
#         two_site2 = np.einsum('XLODPSE,XIJ->LIODJPSE',two_site2,isometry)
# 
#         u1,s1,vt1,dwt = tf.svd(two_site1,two_site1.ndim/2,DMAX)
#         u2,s2,vt2,dwt = tf.svd(two_site2,two_site2.ndim/2,DMAX)
# 
# 
#         # operator to take inverse of s
#         temp = tf.dMult('DM',1./s,isometry)
#         s_inv = np.einsum('Xab,Xcd->acbd',isometry,temp)
# 
#         
#         ### left and right cols ###
#         loop = np.einsum('lidjseLODPSE->lidLODjsePSE',loop)
# 
#         u,s,vt,dwt = tf.svd(loop,loop.ndim/2,2*DMAX)    # (0,0)+(0,1) , (1,0)+(1,1)
#   
#         two_site1 = tf.dMult('MD',u,s)
#         two_site2 = tf.dMult('DM',s,vt)
# 
#         b1dim = len(s)
#         b2dim = int(np.sqrt(b1dim))
# 
#         iden = np.eye(b1dim)[:,:b2dim**2]
#         isometry = iden.reshape(b1dim,b2dim,b2dim)
# 
#         two_site1 = np.einsum('lidLODX,XrR->lirdLORD',two_site1,isometry)
#         two_site2 = np.einsum('XjsePSE,XlL->ljseLPSE',two_site2,isometry)
# 
#         u3,s3,vt3,dwt = tf.svd(two_site1,two_site1.ndim/2,DMAX)
#         u4,s4,vt4,dwt = tf.svd(two_site2,two_site2.ndim/2,DMAX)
# 
#      
# 
#         # tens 00   .... or from u3
#         tens00 = u1.transpose(0,1,2,4,3)
#         lamL = sub_pepx.lambdas[0,0,0]
#         lamI = sub_pepx.lambdas[0,0,1]
#         tens00 = apply_bond(tens00,1./lamL,0)
#         tens00 = apply_bond(tens00,1./lamI,1)
#         try:
#             tens00 = apply_bond(tens00,1./s3,2)
#         except(ValueError):    # pad s with zeros
#             d3 = tens00.shape[2]
#             if d3 > len(s3):                tens00 = tens00[:,:,:len(s3),:]
#             else:                           s3 = s3[:d3]
#         bond00 = [lamL, lamI, s3, s1]   # final 00 bond
# 
#         tens01 = vt1
#         lamI = sub_pepx.lambdas[0,1,1]
#         lamR = sub_pepx.lambdas[0,1,3]
#         tens01 = apply_bond(tens01,1./lamI,1)
#         tens01 = apply_bond(tens01,1./lamR,3)
#         try:
#             tens01 = apply_bond(tens01,1./s4,2)
#         except(ValueError):
#             d4 = tens01.shape[2]
#             if d4 > len(s4):                tens01 = tens01[:,:,:len(s4),:]
#             else:                           s4 = s4[:d4]
#         bond01 = [s1, lamI, s4, lamR]
# 
#         tens10 = u2.transpose(0,1,2,4,3)
#         lamL = sub_pepx.lambdas[1,0,0]
#         lamO = sub_pepx.lambdas[1,0,2]
#         tens10 = apply_bond(tens10,1./lamL,0)
#         tens10 = apply_bond(tens10,1./lamO,2)
#         try:
#             tens10 = apply_bond(tens10,1./s3,1)
#         except(ValueError):
#             tens10 = tens10[:,:,:len(s3),:]
#         bond10 = [lamL, s3, lamO, s2]
# 
#         tens11 = vt2
#         lamO = sub_pepx.lambdas[1,1,2]
#         lamR = sub_pepx.lambdas[1,1,3]
#         tens11 = apply_bond(tens11,1./lamR,3)
#         tens11 = apply_bond(tens11,1./lamO,2)
#         try:
#             tens11 = apply_bond(tens11,1./s4,1)
#         except(ValueError):
#             tens11 = tens11[:,:len(s4),:,:]
#         bond11 = [s2, s4, lamO, lamR]
# 
#         # print 's1',s1
#         # print 's2',s2
#         # print 's3',s3
#         # print 's4',s4
# 
#         new_pepx_gl[0,0] = tens00
#         new_pepx_gl[0,1] = tens01
#         new_pepx_gl[1,0] = tens10
#         new_pepx_gl[1,1] = tens11
# 
#         new_pepx_gl.lambdas[0,0] = bond00
#         new_pepx_gl.lambdas[0,1] = bond01
#         new_pepx_gl.lambdas[1,0] = bond10
#         new_pepx_gl.lambdas[1,1] = bond11
# 
#         # # update vertical bonds by canonicalizing 
#         # vert_gams, vert_lams, axTs = get_sites(new_pepx_gl,(0,0),'o')
#         # canon_gams, canon_lams = canonicalize_GL_list(vert_gams, vert_lams)
#         # new_pepx_gl = set_sites(new_pepx_gl,(0,0),'o',canon_gams,canon_lams,axTs)
# 
#         # vert_gams, vert_lams, axTs = get_sites(new_pepx_gl,(0,1),'o')
#         # canon_gams, canon_lams = canonicalize_GL_list(vert_gams, vert_lams)
#         # new_pepx_gl = set_sites(new_pepx_gl,(0,1),'o',canon_gams,canon_lams,axTs)
# 
#     else:
#         raise(NotImplementedError)
# 
#  
#     return new_pepx_gl   



##################################
#### apply operators to PEPX  ####
##################################


### seems outdated ####
def block_update(gamma_list,lambda_list,block1,block2=None,DMAX=10,canon=0,num_io=1,direction=0,normalize=False):
    ''' apply trotter operator (block) to pepx_list  '''

    new_block = block1.copy()

    pepx_list = GL_to_pepx_list(gamma_list,lambda_list)
    L = len(pepx_list)

    # block * peps
    if np.all([ptens.ndim == 5 for ptens in pepx_list]):   # peps (not reduced)

        peps1 = gamma_list[0]
    
        new_block = np.einsum('AUD...,abcDe->AabcUe...',block1,peps1)
        new_block = tf.reshape(new_block,'ii,...')
    
        ind = 1
        while ind < len(pepx_list):
        
            peps1 = pepxT_list[ind]
        
            p1 = 1 + ind*3 + 1
            p2 = p1 + 2
            p3 = p2 + 3 + (L-ind-1)*2 + 1
        
            indb = range(p1) + range(p2,p2+2) + range(p2+3,p3)
            inds = range(p1-1,p2) + range(p2+1,p2+3)
    
            new_block = np.einsum(new_block,indb,peps1,inds)
        
            ind += 1
        
        new_block = tf.reshape(new_block,'...,ii')
    
        # svd
        lamL, lamR = lambda_list[[0,-1]]
        gam_list, lam_list, errs = tf.decompose_block_GL(new_block,L,DMAX,lamL,lamR,svd_str=5)
    

    # block * pepo
    elif np.all([ptens.ndim == 6 for ptens in pepx_list]):   # pepo

        if block2 is None: 
            pepo1 = pepx_list[0]
    
            new_block = np.einsum('AUD...,abcDde->AabcUde...',block1,pepo1)
            new_block = tf.reshape(new_block,'ii,...')
        
            ind = 1
            while ind < len(pepx_list):

                pepo1 = pepx_list[ind]
            
                p1 = 1 + ind*4 + 1
                p2 = p1 + 2
                p3 = p2 + 4 + (L-ind-1)*2 + 1
        
                indb = range(p1) + range(p2,p2+2) + range(p2+4,p3)
                inds = range(p1-1,p2) + range(p2+1,p2+4)

                # print new_block.shape, indb
                # print pepo1.shape, inds

                new_block = np.einsum(new_block,indb,pepo1,inds)
        
                ind += 1
        
            new_block = tf.reshape(new_block,'...,ii')

        else:    # include tensor dot block2
            pepo1 = pepx_list[0]

            indm1_ = [[0] + [5,6]] + [[i,i+1] for i in range(10,10+(L-1)*4,4)] + [[10+(L-1)*4]]
            inds   = [1,3,4,6,7,9]
            indm2_ = [[2] + [7,8]] + [[i,i+1] for i in range(12,10+(L-1)*4,4)] + [[10+(L-1)*4+1]]

            # flatten lists
            indm1 = [x for y in indm1_ for x in y]
            indm2 = [x for y in indm2_ for x in y]

            new_block = np.einsum(block1,indm1,pepo1,inds,block2,indm2)
            new_block = tf.reshape(new_block,'iii,...')

            ind = 1
            while ind < len(pepx_list):

                pepo1 = pepx_list[ind]
            
                p1 = 1 + ind*4 + 1
                p2 = p1 + 2
                p3 = p2 + 5 + (L-ind-1)*4 + 2
        
                indb = range(p1) + range(p2,p2+2) + range(p2+2,p2+4) + range(p2+5,p3)
                inds = range(p1-1,p2) + range(p2+1,p2+3) + [p2+4]

                new_block = np.einsum(new_block,indb,pepo1,inds)

                ind += 1
        
            new_block = tf.reshape(new_block,'...,iii')

        if normalize:
            norm = np.linalg.norm(new_block)
            new_block = new_block * 1./norm
    
        # svd
        lamL, lamR = lambda_list[0], lambda_list[-1]
        gam_list, lam_list, errs = tf.decompose_block_GL(new_block,L,DMAX,lamL,lamR,svd_str=5)

        # print [m.shape for m in gam_list]
        # print lam_list
       
        # # reorder indices
        # if reorder:     block_list = PEPX.transpose_pepx_list(new_list,axT_invs)
        # else:           block_list = new_list
    

    else:
        raise(TypeError), 'pepx should be either MPS or MPO'


    return gam_list, lam_list, errs

