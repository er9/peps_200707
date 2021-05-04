import numpy as np
import time
import tens_fcts as tf

import MPX
import Operator_2D as Op
import PEP0_env as ENV0
import PEPS_env as ENV
import PEPS_env_SL as ENV_SL


class PEPX(np.ndarray):
    """
    Lightweight MPS/MPO class
    """

    def __new__(cls,tparray,phys_bonds=None):
        """
        Parameters
        ----------
        tparray [array_like]:  nested list of tensors; limited to square lattice
        phys_bonds [2D array of tuples]:  physical bond dimension of each lattice site
        """

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


        if not pepx.ndim == 2:   raise TypeError, 'peps array must be 2D not %d' %pepx.ndim

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
        self.phys_bonds = getattr(pepx, 'phys_bonds',None)  


    def __getitem__(self,item):
        sliced_pepx = super(PEPX,self).__getitem__(item)
        try:
            sliced_pepx.phys_bonds = self.phys_bonds.__getitem__(item)
        except(AttributeError):   
            # np.ndarray has no attribute phys_bonds.  occurs when trying to print? i guess it's called recursively
            pass
        return sliced_pepx

    def __setitem__(self,item,y):
        super(PEPX,self).__setitem__(item,y)
        try:                      # y is also an MPX object
            self.phys_bonds.__setitem__(item,y.phys_bonds)
        except(AttributeError):   # y is just an ndarray
            # print 'setting item'
            # print 'self',self.phys_bonds

            try:
                temp_bonds = y.shape[4:]      # y is an ndarray tens
            except(AttributeError):
                temp_bonds = [yy.shape[4:] for yy in y]   # y is a list of ndarray tens

            # # print self.phys_bonds
            # if isinstance(item,int):       # not sure when this would be called. like an entire row?
            #     temp_bonds = y.shape[4:]
            # elif isinstance(item,tuple):   # single (i,j)
            #     print 'item', item
            #     try:
            #         temp_bonds = y.shape[4:]            # single (i,j)
            #     except(AttributeError):                 # y is list, item is ((is),js)
            #         temp_bonds = [yy.shape[4:] for yy in y]
            #     print temp_bonds
            # else:
            #     ## item is an iterable
            #     print 'item', item         # list of (i,j)'s?
            #     temp_bonds = np.empty(y.shape,dtype=object)
            #     for idx, yy in np.ndenumerate(y):
            #         temp_bonds[idx] = yy.shape[4:]
            #     print temp_bonds

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

    def getSites(mpx1,ind0=(0,0),ns=None):
        if ns is None:  ns = len(mpx1)
        return getSites(mpx1,ind0,ns)

    def transposeUD(pepo):
        return transposeUD(pepo)

    def copy(pepx):
        return copy(pepx)


####################################
#########   PEPX fcts   #############
####################################


##### functions to create PEPX ######

def create(dp, D, fn=np.zeros, split=None, dtype=np.float64):
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
               if j == 0:       pepx[i,j] = fn((1,1,D,D)+dp[i][j],dtype=dtype)   # tensors are lior(ud)
               elif j == L2-1:  pepx[i,j] = fn((D,1,D,1)+dp[i][j],dtype=dtype)
               else:            pepx[i,j] = fn((D,1,D,D)+dp[i][j],dtype=dtype)
            elif i == L1-1:
               if j == 0:       pepx[i,j] = fn((1,D,1,D)+dp[i][j],dtype=dtype) 
               elif j == L2-1:  pepx[i,j] = fn((D,D,1,1)+dp[i][j],dtype=dtype)
               else:            pepx[i,j] = fn((D,D,1,D)+dp[i][j],dtype=dtype)

            elif j == 0:        pepx[i,j] = fn((1,D,D,D)+dp[i][j],dtype=dtype)
            elif j == L2-1:     pepx[i,j] = fn((D,D,D,1)+dp[i][j],dtype=dtype)

            else:               pepx[i,j] = fn((D,D,D,D)+dp[i][j],dtype=dtype)

    return PEPX(pepx, phys_bonds=dp)

def empty(dp, D = None, dtype=np.float64):
    return create(dp, D, fn=np.empty, dtype=dtype)

def zeros(dp, D = None):
    return create(dp, D, fn=np.zeros)

def ones(dp,D = None):
    return create(dp, D, fn=np.ones)

def rand(dp, D = None, seed = None):
    if seed is not None:
        np.random.seed(seed)
    return create(dp, D, fn=np.random.random)

def normalized_rand(dp, D=None, seed=None):

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
    pepx = np.empty((L1,L2),dtype=np.ndarray)

    for idx in np.ndindex((L1,L2)):
        pepx[idx] = np.random.random_sample(shape_array[idx])

    pepx_ = PEPX(pepx)

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


def eye(dp):
    """
    generate identity MPO
    dp:  list of physical bonds (not tuples) at each site
    """

    if not isinstance(dp, np.ndarray):  dp  = np.array(dp)
    L1,L2 = dp.shape
    # L1,L2 = len(dp),len(dp[0])

    id_pepo = np.empty((L1,L2),dtype=np.object)
    for i in range(L1):
        for j in range(L2):
            id_pepo[i,j] = np.eye(dp[i,j]).reshape(1,1,1,1,dp[i,j],dp[i,j])

    return PEPX(id_pepo)


def copy(pepx):
    # deeper copy than mpx.copy, as it copies the ndarrays in mpx
    new_pepx = empty(pepx.phys_bonds,1)
    for idx, tens in np.ndenumerate(pepx):
        new_pepx[idx] = tens.copy()
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
    
    return ENV0.contract(mats)


def conj_transpose(pepx):

    if   pepx[0,0].ndim == 5:  return np.conj(pepx)
    elif pepx[0,0].ndim == 6:  
        
        pepx_ = pepx.copy()
        for i in np.ndenumerate(pepx_):
            pepx_[i] = np.conj(pepx_[i].transpose([0,1,2,3,5,4]))
        return pepx_


def flatten(pepx):
    """   Converts PEPO object into PEPS    """
    peps = np.empty(pepx.shape,dtype=object)
    for ind in np.ndindex(pepx.shape):
        ptens = pepx[ind]
        if ptens.ndim == 5:  peps[ind] = ptens
        else:                peps[ind] = tf.reshape(ptens,'i,i,i,i,...',group_ellipsis=True)
    return PEPX(peps)


def unflatten(pepx,dbs):
    '''  converts PEPS object into PEPO '''
    pepo = np.empty(pepx.shape,dtype=object)
    for ind in np.ndindex(pepx.shape):
        ptens = pepx[ind]
        pepo[ind] = ptens.reshape( ptens.shape[:4]+dbs[ind] )
    return PEPX(pepo,dbs)


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
    

##############################
#### arithemtic functions ####
##############################

# @profile
def vdot(pepx1, pepx2, side='I',XMAX=100,contract_SL=False,scaleX=1):
    """
    vdot of two PEPS <psi1|psi2> (or PEPO-->PEPS). returns scalar/squeezed tensor

    cf. np.vdot
    """
    # try:
    #     return mps_dot(np.conj(mps1), mps2, direction)
    # except(ValueError):

    ovlp =  peps_dot(np.conj(flatten(pepx1)),flatten(pepx2),side=side,XMAX=XMAX,
                     contract_SL=contract_SL,scaleX=scaleX)
    return np.squeeze(ovlp)


# @profile
def peps_dot(peps1, peps2, side='I',XMAX=100,contract_SL=False,scaleX=1):
    """
    dot of two PEPS, returns tensor
    """
    L1, L2 = peps1.shape
    assert peps1.shape == peps2.shape, 'mps_dot: peps1, peps2 need to be same size'
    
    if contract_SL:
        ovlp = ENV_SL.get_ovlp(peps1,peps2,side=side,XMAX=XMAX,scaleX=scaleX)
    else:
        ovlp = ENV.get_ovlp(peps1,peps2,side=side,XMAX=XMAX)
    return ovlp


# @profile
def norm(pepx,side='I',XMAX=100,contract_SL=False,scaleX=1): 
    """
    2nd norm of a MPX
    """
    norm_tens = peps_dot(np.conj(flatten(pepx)),flatten(pepx),side=side,XMAX=XMAX,
                         contract_SL=contract_SL,scaleX=scaleX)

    norm_val = np.einsum('ee->',norm_tens)
    return np.sqrt(norm_val)


def trace_norm(pepx,side='I',XMAX=100):
    ''' trace norm of PEPX '''

    tr_pep0 = ENV0.trace(pepx)

    norm1 = ENV0.contract(tr_pep0, 'I', XMAX)
    norm2 = ENV0.contract(tr_pep0, 'R', XMAX)
    norm3 = ENV0.contract(tr_pep0, 'L', XMAX)
    norm4 = ENV0.contract(tr_pep0, 'O', XMAX)
    
    diffs = np.abs([norm1-norm2,norm1-norm3,norm1-norm4,norm2-norm3,norm3-norm4])
    if np.any(diffs > 1.0e-8):
        raise RuntimeError('PEPX: trace norm err %8.6e'%diffs)


    return ENV0.contract(tr_pep0, side, XMAX)


def mul(alpha, pepx):
    """
    scales mpx by constant factor alpha (can be complex)
    """

    Ls = pepx.shape                            # can work with high-dimensional arrays
    new_pepx = np.empty(Ls,dtype=np.object)

    const = np.abs(alpha)**(1./np.prod(Ls))
    dtype = np.result_type(alpha,pepx[0,0])
    for idx, m in np.ndenumerate(pepx):
        new_pepx[idx] = np.array(m,dtype=dtype)*const

    # change sign as specified by alpha
    if dtype == int or dtype == float:
        phase = np.sign(alpha)
    elif dtype == complex:
        phase = np.exp(1j*np.angle(alpha))
    else:
        raise(TypeError), 'not valid datatype %s' %dtype

    new_pepx[0,0] *= phase

    return PEPX(new_pepx,pepx.phys_bonds)


### doesn't actually work ###
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
    dtype = np.result_type(pepx1[0,0], pepx2[0,0])

    for i in np.ndindex((L1,L2)):
        sh1 = pepx1[i].shape
        sh2 = pepx2[i].shape

        l1,u1,d1,r1 = sh1[:4]
        l2,u2,d2,r2 = sh2[:4]
        dp_sh = sh1[4:]

        new_site = np.zeros((l1+l2,u1+u2,d1+d2,r1+r2)+dp_sh,dtype=dtype)
        new_site[:l1,:u1,:d1,:r1] = pepx1[i].copy()
        new_site[l1:,u1:,d1:,r1:] = pepx2[i].copy()

        new_pepx[i] = new_site.copy()


    if obc[0]:   # left boundary
        for i in range(L1):
            l1 = pepx1[i,0].shape[0]
            l2 = pepx2[i,0].shape[0]
 
            if l1 == l2:
                sh3 = new_pepx[i,0].shape
                new_site = new_pepx[i,0][:l1,:,:,:] + new_pepx[i,0][l1:,:,:,:]
                new_site = new_site.reshape((l1,)+sh3[1:])
                new_pepx[i,0] = new_site.copy()

    if obc[1]:   # upper boundary
        for i in range(L2):
            i1 = pepx1[0,i].shape[1]
            i2 = pepx2[0,i].shape[1]
          
            if i1 == i2:  
                sh3 = new_pepx[0,i].shape
                new_site = new_pepx[0,i][:,:i1,:,:] + new_pepx[0,i][:,i1:,:,:]
                new_site = new_site.reshape(sh3[:1]+(i1,)+sh3[2:])
                new_pepx[0,i] = new_site.copy()

    if obc[2]:   # lower boundary
        for i in range(L2):
            o1 = pepx1[-1,i].shape[2]
            o2 = pepx2[-1,i].shape[2]
          
            if o1 == o2:  
                sh3 = new_pepx[-1,i].shape
                new_site = new_pepx[-1,i][:,:,:o1,:] + new_pepx[-1,i][:,:,o1:,:]
                new_site = new_site.reshape(sh3[:2]+(o1,)+sh3[3:])
                new_pepx[-1,i] = new_site.copy()
    if obc[3]:   # right boundary
        for i in range(L1):
            r1 = pepx1[i,-1].shape[3]
            r2 = pepx2[i,-1].shape[3]

            if r1 == r2:
                sh3 = new_pepx[i,-1].shape
                new_site = new_pepx[i,-1][:,:,:,:r1] + new_pepx[i,-1][:,:,:,r1:]
                new_site = new_site.reshape(sh3[:3]+(r1,)+sh3[4:])
                new_pepx[i,-1] = new_site.copy()

    return PEPX(new_pepx)



def add_el(pepx1, pepx2):
    """
    Elemental addition
    """
    assert pepx1.shape==pepx2.shape, 'add el:  mpx1 and mpx2 need to have same shape'

    new_pepx = empty(pepx1.phys_bonds)
    for i in np.ndenumerate(pepx1):
        assert(pepx1[i].shape == pepx2[i].shape), 'add el:  mpx1, mpx2 need to have same shape'
        new_pepx[i] = pepx1[i]+pepx2[i]
        
    return new_pepx


def axpby(alpha,pepx1,beta,pepx2):
    """
    return (alpha * mpx1) + (beta * mpx2)
    alpha, beta are scalar; mps1,mps2 are ndarrays of tensors (MPXs)
    """

    pepx_new = add(mul(alpha,pepx1),mul(beta,pepx))
    return pepx_new


# @profile
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
    assert pepx2.shape==Ls, '[dot]: shapes of pepx1 and pepx2 are not equal'
    new_pepx = np.empty(Ls, dtype=np.object)

    # if np.all([ pepx1[i].ndim==3 and pepx2[i].ndim==3 for i in np.ndenumerate(pepx1) ]):
    #     return peps_dot(pepx1,pepx2)
    # else:

    # print 'dot'
    for i in np.ndindex(Ls):
        len_dp1 = len(pepx1.phys_bonds[i])
        len_dp2 = len(pepx2.phys_bonds[i])
        ax1 = [0,2,4,6] + range(8, 8+len_dp1)
        ax2 = [1,3,5,7] + range(8+len_dp1-1,8+len_dp1+len_dp2-1)
        ax2[-len_dp2] = ax1[-1]   # contract vertical bonds (mpx1 down with mpx2 up)a
        # print pepx1[i].shape, ax1
        # print pepx2[i].shape, ax2
        new_site = np.einsum(pepx1[i],ax1,pepx2[i],ax2)
        new_pepx[i] = tf.reshape(new_site,'ii,ii,ii,ii,...',group_ellipsis=False)
 

    return PEPX(new_pepx)



def outer(pepx_u, pepx_d):
    ''' take outer product of pepx_u, pepx_d '''
    Ls = pepx_u.shape

    out = np.empty(Ls,dtype=np.object)
    for idx in np.ndindex(Ls):
        tens = np.einsum('lioru,LIORd->lLiIoOrRud',pepx_u[idx],pepx_d[idx])
        out[idx] = tf.reshape(tens,'ii,ii,ii,ii,i,i')
        # print tens.shape

    return PEPX(out)


#################################################
######### measure op expectation vals ###########
#################################################

def meas_obs(pepx,pepo,ind0=(0,0),op_conn=None,pepx_type='peps',side='I',XMAX=100,envs=None,return_norm=False,
             contract_SL=False,scaleX=1):
    # operator acts on sites ind0 : ind0+len(mpo)
    # pepo is not actually a pepo--any form of the following operators

    # measure magnetization
    if isinstance(pepo,str):
        return meas_obs_mag(pepx,pauli_op=pepo,bounds=envs,XMAX=XMAX,return_norm=return_norm,
                            contract_SL=contract_SL,scaleX=scaleX)

    # measure correlations will row/col/all sites with site at ind0 
    elif isinstance(pepo,tuple):
        try:     ind0 = pepo[2] # pauli operators
        except:  ind0 = None    # defaults to L1/2, L2/2

        return meas_obs_corr(pepx,pauli_op=pepo[:2],ref_ind=ind0,bounds=envs,XMAX=XMAX,return_norm=return_norm,
                             contract_SL=contract_SL,scaleX=scaleX)

    # measure list of single site operators acting simultaneously
    elif isinstance(pepo,list):
        ops = pepo[0]
        idxs = pepo[1]
        return meas_obs_prod_op(pepx,ops,idxs,bounds=envs,XMAX=XMAX,return_norm=return_norm,
                                contract_SL=contract_SL,scaleX=scaleX)

    elif isinstance(pepo,Op.TrotterOp):
        if pepx[0,0].ndim == 5:
            expVal = meas_obs_trotterH(pepx,pepo,XMAX=XMAX,bounds=envs,return_norm=return_norm,
                                       contract_SL=contract_SL,scaleX=scaleX)
        elif pepx[0,0].ndim == 6:
            expVal = meas_obs_trotterH_rho(pepx,pepo,XMAX=XMAX,bounds=envs,return_norm=return_norm)
        return expVal

    elif pepo is None:
        exp_val = np.nan
        if return_norm:
            norm_val = norm(pepx)
            return exp_val, norm_val
        else:
            return exp_val

    else:
        if isinstance(pepo,PEPX):
            L1,L2 = pepo.shape
            ix,iy = ind0
            pepx_op = pepx.copy()
            pepx_op[ix:ix+L1,iy:iy+L2] = dot(pepo,pepx[ix:ix+L1,iy:iy+L2])

        elif isinstance(pepo,MPX.MPX):
            xs, ys = get_conn_inds(op_conn)
            app_list, errs = mpo_update(pepx[xs,ys],op_conn,pepo,DMAX=-1,regularize=False)
            pepx_op = pepx.copy()
            pepx_op[xs,ys] = app_list

        if envs is not None:
            exp_val = ENV.embed_sites_ovlp(flatten(np.conj(pepx)),flatten(pepx_op),envs)
        else:
            if pepx_type in ['peps']:
                exp_val = vdot(pepx,pepx_op,side=side,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
            elif pepx_type in ['dm','DM','rho','pepo']:
                trPEP0 = ENV0.trace(pepx_op)
                exp_val = ENV0.contract(trPEP0,XMAX=XMAX)
            else:
                raise(NotImplementedError),'mpx meas obs not implemented for pepx type %s'%s

        if return_norm:
            norm_val = norm(pepx,contract_SL=contract_SL)
            exp_val = exp_val/(norm_val**2)
            return exp_val, norm_val
        else:
            return exp_val
   

## get bounds as needed
def meas_get_bounds(pepx,bounds,op_shape,XMAX,contract_SL,scaleX=1):

    L1, L2 = pepx.shape
    NR, NC = op_shape

    # set which contraction scheme is used
    if contract_SL:
        return meas_get_bounds_SL(pepx,bounds,op_shape,XMAX,contract_SL,scaleX)

    # else...
    get_bounds = ENV.get_boundaries
    get_subbounds = ENV.get_subboundaries

    # calculate envs and sweep through
    if bounds is None:
        envIs = get_bounds( np.conj(pepx), pepx, 'I', L1, XMAX=XMAX)  # list of len L+1
        envOs = get_bounds( np.conj(pepx), pepx, 'O', 0 , XMAX=XMAX)  # list of len L+1

        senvRs = []
        senvLs = []
        if NR > 0:
            for i in range(L1):
                NR_ = min(NR,L1-i)
                senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],
                                           np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'R',0 ,XMAX=XMAX))
                senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],
                                           np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'L',L2,XMAX=XMAX))
    else:
        senvLs, envIs, envOs, senvRs = bounds

        if NR > 0:
            if senvLs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(pepx[i:i+NR_,:]),
                                                pepx[i:i+NR_,:],'R',0 ,XMAX=XMAX))

            if senvRs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(pepx[i:i+NR_,:]),
                                                pepx[i:i+NR_,:],'L',L2,XMAX=XMAX))

    return senvLs, envIs, envOs, senvRs


def meas_get_bounds_SL(pepx,bounds,op_shape,XMAX,contract_SL,scaleX):

    L1, L2 = pepx.shape
    NR, NC = op_shape

    # set which contraction scheme is used
    get_bounds = ENV_SL.get_boundaries
    get_subbounds = ENV_SL.get_subboundaries

    # calculate envs and sweep through
    if bounds is None:
        envIs = get_bounds( np.conj(pepx), pepx, 'I', L1, XMAX=XMAX,scaleX=scaleX)  # list of len L+1
        envOs = get_bounds( np.conj(pepx), pepx, 'O', 0 , XMAX=XMAX,scaleX=scaleX)  # list of len L+1

        senvRs = []
        senvLs = []
        if NR > 0:
            for i in range(L1):
                NR_ = min(NR,L1-i)
                senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],
                                           np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'R',0 ,XMAX=XMAX,scaleX=scaleX))
                senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],
                                           np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'L',L2,XMAX=XMAX,scaleX=scaleX))
    else:
        senvLs, envIs, envOs, senvRs = bounds

        if NR > 0:
            if senvLs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvRs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(pepx[i:i+NR_,:]),
                                                pepx[i:i+NR_,:],'R',0 ,XMAX=XMAX,scaleX=scaleX))

            if senvRs is None:
                for i in range(L1):
                    NR_ = min(NR,L1-i)
                    senvLs.append(get_subbounds(envIs[i],envOs[i+NR_],np.conj(pepx[i:i+NR_,:]),
                                                pepx[i:i+NR_,:],'L',L2,XMAX=XMAX,scaleX=scaleX))


    # put into DL form
    tempIs, tempOs = [], []
    for i in range(len(envIs)):
        tempIs.append(ENV_SL.SL_to_DL_bound(envIs[i],'row'))
        tempOs.append(ENV_SL.SL_to_DL_bound(envOs[i],'row'))

    tempLs, tempRs = [], []
    for j1 in range(len(senvLs)):
        subLs, subRs = [], []
        for j2 in range(len(senvLs[j1])):
            subLs.append(ENV_SL.SL_to_DL_bound(senvLs[j1][j2],'col'))
            subRs.append(ENV_SL.SL_to_DL_bound(senvRs[j1][j2],'col'))
        tempLs.append([bm for bm in subLs])
        tempRs.append([bm for bm in subRs])

    return tempLs, tempIs, tempOs, tempRs


# @profile
def meas_obs_trotterH(pepx,trotterH,XMAX=100,bounds=None,return_norm=False,contract_SL=False,scaleX=1):

    L1, L2 = trotterH.Ls
    NR, NC = trotterH.it_sh

    # print 'meas H pepx', contract_SL,XMAX,scaleX

    senvLs, envIs, envOs, senvRs = meas_get_bounds(pepx,bounds,(NR,NC),XMAX,contract_SL,scaleX)

     
    op_inds = {}            # elements to call for each distinct op_conn
    for opk in trotterH.ops.keys():
        try:
            xx = op_inds[trotterH.conn[opk]]
        except(KeyError):
            inds_list = get_conn_inds(trotterH.conn[opk],trotterH.ind0[opk])
            op_inds[trotterH.conn[opk]] = inds_list


    obs_val = 0.
    for idx, m_op in trotterH.get_trotter_list():

        i,j = idx
        NR_ = min(NR,L1-i)
        NC_ = min(NC,L2-j)

        sub_pepx = pepx[i:i+NR_,j:j+NC_]

        bi = envIs[i][j:j+NC_]
        bo = envOs[i+NR_][j:j+NC_]

        bl = senvLs[i][j]
        br = senvRs[i][j+NC_] 

        xs, ys = op_inds[trotterH.conn[m_op]]
        pepx_list, axTs = connect_pepx_list(sub_pepx[xs,ys], trotterH.conn[m_op])
        app_list, errs = mpo_update(pepx_list,None,trotterH.ops[m_op],DMAX=-1,regularize=False)

        app_pepx = sub_pepx.copy()
        app_pepx[xs,ys] = transpose_pepx_list(app_list, axTs)
        
        exp_val = ENV.embed_sites_ovlp(np.conj(sub_pepx),app_pepx,[bl,bi,bo,br],XMAX=XMAX)

        # exp_val = meas_obs(sub_pepx,trotterH.ops[m_op],ind0=trotterH.ind0[m_op],op_conn=trotterH.conn[m_op],
        #                    envs=[bl,bi,bo,br])
        
        obs_val += exp_val

    if return_norm:
    
        norm_val = np.sqrt( ENV.ovlp_from_bound(envIs[L1]) )
        obs_val = obs_val/(norm_val**2)
        return obs_val, norm_val

    else:

        return obs_val


# def meas_obs_trotterH_rho(pepx,trotterH,bounds=None,XMAX=100,return_norm=False):
#     # pepx could also be pepo ('dm','DM','rho','pepo')
# 
#     L1, L2 = trotterH.Ls
#     NR, NC = trotterH.it_sh
# 
#     trPEPO = ENV0.trace(pepx)
# 
#     # calculate envs and sweep through
#     if bounds is None:
#         envIs = ENV0.get_boundaries( trPEPO, 'I', L1, XMAX=XMAX)  # list of len ii+1
#         envOs = ENV0.get_boundaries( trPEPO, 'O', 0 , XMAX=XMAX)  # list of len L+1
# 
#         senvRs = []
#         senvLs = []
#         for i in range(L1):
#             NR_ = min(NR,L1-i)
#             senvRs.append(ENV0.get_subboundaries(envIs[i],envOs[i+NR_],trPEPO[i:i+NR_,:],'R',0 ,XMAX=XMAX))
#             senvLs.append(ENV0.get_subboundaries(envIs[i],envOs[i+NR_],trPEPO[i:i+NR_,:],'L',L2,XMAX=XMAX))
#     else:
#         senvLs, envIs, envOs, senvRs = bounds
# 
#      
#     op_inds = {}            # elements to call for each distinct op_conn
#     for opk in trotterH.ops.keys():
#         try:
#             xx = op_inds[trotterH.conn[opk]]
#         except(KeyError):
#             inds_list = get_conn_inds(trotterH.conn[opk],trotterH.ind0[opk])
#             op_inds[trotterH.conn[opk]] = inds_list
#             # print self.conn[opk], inds_list
# 
#     obs_val = 0.
#     for idx, m_op in trotterH.get_trotter_list():
#         i,j = idx
#         NR_ = min(NR,L1-i)
#         NC_ = min(NC,L2-j)
# 
#         sub_pepx = pepx[i:i+NR_,j:j+NC_]
# 
#         bi = envIs[i][j:j+NC_]
#         bo = envOs[i+NR_][j:j+NC_]
# 
#         bl = senvLs[i][j]
#         br = senvRs[i][j+NC_] 
# 
#         xs, ys = op_inds[trotterH.conn[m_op]]
#         pepx_list, axTs = connect_pepx_list(sub_pepx[xs,ys], trotterH.conn[m_op])
#         app_list, errs = mpo_update(pepx_list,None,trotterH.ops[m_op],DMAX=XMAX)
# 
#         app_pepx = sub_pepx.copy()
#         app_pepx[xs,ys] = transpose_pepx_list(app_list, axTs)
#         
#         exp_val  = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='I',XMAX=XMAX)
#         # exp_val1 = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='O',XMAX=XMAX)
#         # exp_val2 = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='L',XMAX=XMAX)
#         # exp_val3 = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='R',XMAX=XMAX)
# 
#         # diffs = np.abs([exp_val-exp_val1, exp_val-exp_val2, exp_val-exp_val3, exp_val1-exp_val2, exp_val1-exp_val3, exp_val2-exp_val3])
#         # print diffs
#         # if np.any(diffs > 1.0e-8):
#         #    print 'embed sites error'
#         #    raise RuntimeError('embed contract direction inconsistent')
# 
#         obs_val += exp_val
# 
#     if return_norm:
#     
#         norm_val = np.sqrt( ENV.ovlp_from_bound(envIs[L1]) )
#         obs_val = obs_val/(norm_val**2)
#         return obs_val, norm_val
# 
#     else:
# 
#         return obs_val


def meas_obs_mag(pepx,pauli_op='SZ',bounds=None,XMAX=100,return_norm=False,return_all=False,contract_SL=False,
                 scaleX=1):
    # pepx could also be pepo ('dm','DM','rho','pepo')

    L1, L2 = pepx.shape
    NR, NC = (1,1)

    senvLs, envIs, envOs, senvRs = meas_get_bounds(pepx,bounds,(NR,NC),XMAX,contract_SL,scaleX)

    # get exp val
    op = Op.paulis[pauli_op]

    if return_all:    obs_val = np.empty((L1,L2))
    else:             obs_val = 0.

    for idx in np.ndindex(L1,L2):

        i,j = idx
        NR_ = min(NR,L1-i)   # = 1
        NC_ = min(NC,L2-j)   # = 1

        sub_pepx = pepx[i:i+1,j:j+1]

        bi = envIs[i][j:j+1]
        bo = envOs[i+NR_][j:j+1]

        bl = senvLs[i][j]
        br = senvRs[i][j+1] 

        app_pepx = sub_pepx.copy()
        app_pepx[0,0] = np.einsum('liord,Dd->liorD',app_pepx[0,0],op)
        exp_val = ENV.embed_sites_ovlp(np.conj(sub_pepx),app_pepx,[bl,bi,bo,br],XMAX=XMAX)
        
        if return_all:  obs_val[idx] = exp_val
        else:           obs_val += exp_val

    if return_norm:    
        norm_val = np.sqrt( ENV.ovlp_from_bound(envIs[L1]) )
        obs_val = obs_val/(norm_val**2)
        return obs_val, norm_val
    else:
        return obs_val


# def meas_obs_1site(pepx,ops,idxs,envs_list=None,XMAX=100,return_norm=False):
# 
#     app_pepx = pepx.copy()
#     for i in range(len(ops)):
#         app_pepx[idxs[i]] = np.einsum('liord,Dd->liorD',app_pepx[idxs[i]],ops[i])
# 
#     if envs_list is None:
#         obs_val = vdot(pepx,app_pepx,XMAX=XMAX)
#     else: 
#         obs_val = ENV.embed_site_ovlp(np.conj(pepx),app_pepx,envs_list,XMAX=XMAX)
# 
#     return obs_val


# def meas_obs_prod_op(pepx,ops,idxs,envs_list=None,XMAX=100,return_norm=False,contract_SL=False):
# 
#     app_pepx = pepx.copy()
#     for i in range(len(ops)):
# 
#         try:                # ops[i] = 'SX', 'SY', 'SZ', or 'ID'
#             op_mat = Op.paulis[ops[i]]
#         except(KeyError):   # ops[i] should be an operator (2D matrix)
#             op_mat = ops[i]
# 
#         app_pepx[idxs[i]] = np.einsum('liord,Dd->liorD',app_pepx[idxs[i]],op_mat)
# 
#     if envs_list is None:
#         obs_val = vdot(pepx,app_pepx,XMAX=XMAX,contract_SL=contract_SL)
#         if return_norm:  norm_val = norm(pepx,XMAX=XMAX,contract_SL=contract_SL)
#     else: 
#         obs_val = ENV.embed_site_ovlp(np.conj(pepx),app_pepx,envs_list,XMAX=XMAX)
#         if return_norm:  norm_val = ENV.embed_site_norm(pepx,envs_list,XMAX=XMAX)
# 
#     if return_norm:
#         return obs_val/(norm_val**2), norm_val
#     else:
#         return obs_val


def meas_obs_prod_op(pepx,ops,idxs,bounds=None,XMAX=100,return_norm=False,contract_SL=False,scaleX=1):

    app_pepx = pepx.copy()
    min_r, max_r = 0,0 

    # bounds=None

    for i in range(len(ops)):

        if idxs[i][0] < min_r:   min_r = idxs[i][0]
        if idxs[i][0] > max_r:   max_r = idxs[i][0]

        try:                # ops[i] = 'SX', 'SY', 'SZ', or 'ID'
            op_mat = Op.paulis[ops[i]]
        except(KeyError):   # ops[i] should be an operator (2D matrix)
            op_mat = ops[i]

        app_pepx[idxs[i]] = np.einsum('liord,Dd->liorD',app_pepx[idxs[i]],op_mat)

    if bounds is None:
        obs_val = vdot(pepx,app_pepx,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
        if return_norm:  norm_val = norm(pepx,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
    else:
        senvLs, envIs, envOs, senvRs = bounds
        sbL = MPX.ones([(1,1)]*(max_r-min_r+1))
        sbR = MPX.ones([(1,1)]*(max_r-min_r+1))
        envs_list = [sbL, envIs[min_r], envOs[max_r+1], sbR]

        # print 'idxs', idxs
        # print [m.shape for m in sbL], [m.shape for m in envIs[min_r]]
        # print [m.shape for m in sbR], [m.shape for m in envOs[max_r+1]]
       
        obs_val = ENV.get_sub_ovlp(np.conj(pepx[min_r:max_r+1,:]), app_pepx[min_r:max_r+1,:], envs_list,
                                   side='L', XMAX=XMAX)
        if return_norm:  norm_val = np.sqrt( ENV.get_sub_ovlp(np.conj(pepx[min_r:max_r+1,:]),pepx[min_r:max_r+1,:],
                                                              envs_list,side='L',XMAX=XMAX) )
        # obs_val = ENV.embed_site_ovlp(np.conj(pepx),app_pepx,envs_list,XMAX=XMAX)
        # if return_norm:  norm_val = ENV.embed_site_norm(pepx,envs_list,XMAX=XMAX)

    if return_norm:
        return obs_val/(norm_val**2), norm_val
    else:
        return obs_val

            
def meas_obs_corr(pepx,pauli_op=['SZ','SZ'],ref_ind=None,axis=None,bounds=None,XMAX=100,return_norm=False,
                  contract_SL=False,scaleX=1):
    # pepx could also be pepo ('dm','DM','rho','pepo')

    L1, L2 = pepx.shape

    # get exp val
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
        pepx_ = pepx.copy()

        i,j = idx
        pepx_[i0,j0] = np.einsum('liord,Dd->liorD',pepx_[i0,j0],op1)
        pepx_[i ,j ] = np.einsum('liord,Dd->liorD',pepx_[i ,j ],op2)

        exp_val = vdot(pepx,pepx_,XMAX=XMAX,contract_SL=contract_SL,scaleX=scaleX)
        
        obs_val.append( exp_val )

    if return_norm:
    
        norm_val = np.sqrt( norm(pepx,contract_SL=contract_SL,scaleX=scaleX) )
        obs_val = [ov/(norm_val**2) for ov in obs_val]
        return obs_val, norm_val

    else:

        return obs_val




############################################
######## fcts for working with PEPX  #######
############################################ 


def get_io_axT(in_legs,out_legs,ndim,d_side='R',d_end=False):
    ''' iso_legs:  str or list of 'l','u','d','r':  legs to be on the grouped with io axes and placed on right
        iso_side = right: (not iso),(iso + io)
        iso_side = left:  (iso + io),(not iso)
        ndim = dimension of pepx tensor
        d_end = True:  phys_bond is always in the last position
    '''
 
    io_ax  = range(4,ndim)
   
    axL = []
    axM = []
    axR = []
    
    for ind in range(4):
        if   'lior'[ind] in in_legs:     axL.append(ind)   # left side of matrix
        elif 'lior'[ind] in out_legs:    axR.append(ind)   # right side of matrix
        else:                            axM.append(ind)

    if d_end:
        axT = axL + axM + axR + io_ax
    else:
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

    axT, axT_inv = get_io_axT(iso_legs,iso_side, pepx_tens.ndim)
   
    if   iso_side in ['r','R','o','O','right']:
        axT, axT_inv = get_io_axT('', iso_legs, pepx_tens.ndim)
        tens = pepx_tens.transpose(axT)
        split_ind = -1*( len(iso_legs) + pepx_tens.ndim-4 )
    elif iso_side in ['l','L','i','I','left']:
        axT, axT_inv = get_io_axT(iso_legs, '', pepx_tens.ndim)
        tens = pepx_tens.transpose(axT)
        split_ind = len(iso_legs) + pepx_tens.ndim-4

    mat = tf.reshape(tens,split_ind)
    return mat, tens.shape, axT_inv


def unmatricize(peps_mat,tens_sh,axT_inv):
    '''  peps_mat is matrix (..., iso_legs + io)
         tens_sh:  shape of re-ordered tensor before reshaping
    '''
    tens = peps_mat.reshape(tens_sh).transpose(axT_inv)
    return tens


def opposite_leg(leg):
    op_leg = ''
    for xx in leg:
        if   xx == 'l':  op_leg += 'r'
        elif xx == 'r':  op_leg += 'l'
        elif xx == 'o':  op_leg += 'i'
        elif xx == 'i':  op_leg += 'o'
        elif xx == 'u':  op_leg += 'd'
        elif xx == 'd':  op_leg += 'u'

    return op_leg


def leg2ind(leg):
    if   isinstance(leg,str):
        if   leg == 'l':   return 0
        elif leg == 'i':   return 1
        elif leg == 'o':   return 2
        elif leg == 'r':   return 3 
        elif leg == 'u':   return 4
        elif leg == 'd':   return 5  
    elif isinstance(leg,int):
        if   leg == 0:     return 'l'
        elif leg == 1:     return 'i'
        elif leg == 2:     return 'o'
        elif leg == 3:     return 'r'
        elif leg == 4:     return 'u'
        elif leg == 5:     return 'd'


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


# def get_conn_inds(op_conn):
#     inds_list = []
#     for x in op_conn:
#         if   x in ['r','R']:   inds_list.append((0,1))
#         elif x in ['l','L']:   inds_list.append((0,-1))
#         elif x in ['i','I']:   inds_list.append((-1,0))
#         elif x in ['o','O']:   inds_list.append((1,0))
#         else:                  inds_list.append(())
#     return inds_list

def get_conn_inds(op_conn,ind0=(0,0)):
    ''' connectivity str -> [xs,ys] '''
    xs = (ind0[0],)
    ys = (ind0[1],)
    for x in op_conn:
        if   x in ['r','R']:
            xs = xs + (xs[-1],)
            ys = ys + (ys[-1]+1,)
        elif x in ['l','L']:
            xs = xs + (xs[-1],)
            ys = ys + (ys[-1]-1,)
        elif x in ['i','I']:
            xs = xs + (xs[-1]-1,)
            ys = ys + (ys[-1],)
        elif x in ['o','O']:
            xs = xs + (xs[-1]+1,)
            ys = ys + (ys[-1],)
    return [xs,ys]


def get_inds_conn(inds_list,wrap_inds=False):
    ''' get connection direction as progress in inds_list
        wrap_inds:  also return connection direction from last to first in list
        inds_list: [xs,ys]
    '''
    op_conns = ''

    for i in range(len(inds_list[0])-1):
        dx = inds_list[0][i+1]-inds_list[0][i]
        dy = inds_list[1][i+1]-inds_list[1][i]
        if   (dx,dy) == (0,1):   op_conns += 'r'
        elif (dx,dy) == (0,-1):  op_conns += 'l'
        elif (dx,dy) == (-1,0):  op_conns += 'i'
        elif (dx,dy) == (1,0):   op_conns += 'o'
        else:
            raise IndexError('PEPX: get_inds_conn inds_list not connected')

    if wrap_inds:
        x0,y0 = (inds_list[0][0],  inds_list[1][0])
        xL,yL = (inds_list[0][-1], inds_list[1][-1])
        op_conns += get_inds_conn([[xL,x0],[yL,y0]])
        
    return op_conns


def get_sites(pepx,ind0,connect_list,d_side='R'):

    xs,ys = get_conn_inds(connect_list,ind0)
    site_list, axT_invs =  connect_pepx_list(pepx[xs,ys],connect_list,side=d_side)

    return site_list, axT_invs


def set_sites(pepx,peps_list,ind0,connect_list,axT_invs):

    pepx_ = pepx.copy()

    xs,ys = get_conn_inds(connect_list,ind0)
    pepxT_list = transpose_pepx_list(peps_list,axT_invs)
    
    pepx_[xs,ys] = pepxT_list
    return pepx_
    

   
def QR_factor(pepx_tens,iso, d_end=False):

    # mat, mat_sh, axT_inv = matricize(pepx_tens,iso,'r')    
    # Q, R = np.linalg.qr(mat)
    # Q = Q.reshape(mat_sh[:-1]+(-1,))
    # R = R.reshape((-1,)+mat_sh[1])

    axT, axT_inv = get_io_axT('',iso,pepx_tens.ndim,d_end=d_end)
    # print 'qr factor', axT, axT_inv
    tens = pepx_tens.transpose(axT)

    Q, R = tf.qr(tens,4-len(iso))
    # print tens.shape, Q.shape, R.shape
  
    # if d_end:
    #     num_io = pepx_tens.ndim-4
    #     for x in range(num_io):
    #         # print 'QR axT', axT, axT_inv
    #         R = np.moveaxis(R,1,-1)
    #         axT_new = np.delete(np.append(axT,axT[-2]),-3)
    #         axT = axT_new
    #     axT_inv = np.argsort(axT)

    # print 'QR', pepx_tens.shape, iso, Q.shape, R.shape
    # print 'QR factor', d_end,iso, Q.shape, R.shape, axT_inv

    return Q, R, axT_inv
 
   
def LQ_factor(pepx_tens,iso, d_end=False):

    # mat, mat_sh, axT_inv = matricize(pepx_tens,iso,'l')
    # Q, R = np.linalg.qr(mat.T)
    # L = (R.T).reshape(mat_sh[0]+(-1,))
    # Q = (Q.T).reshape((-1,)+mat_sh[1])

    axT, axT_inv = get_io_axT(iso,'',pepx_tens.ndim,d_side='L',d_end=d_end)
    tens = pepx_tens.transpose(axT)

    io_dim = pepx_tens.ndim-4
    L, Q = tf.lq(tens,len(iso)+io_dim)

    # if d_end:
    #     num_io = pepx_tens.ndim-4
    #     L = np.moveaxis(L,-1,-1-num_io)

    # print 'LQ', d_end, iso, L.shape, Q.shape, axT_inv

    return L, Q, axT_inv


def QR_contract(Q,R,axT_inv,d_end=False):
    # d_end:  don't need to do anything bc axT_inv takes care of it

    # print 'QR', Q.shape, R.shape, axT_inv

    tens = np.tensordot(Q,R,axes=(-1,0))
    return tens.transpose(axT_inv)
    

def LQ_contract(L,Q,axT_inv,d_end=False):
 
    if d_end: 
        num_io = L.ndim + Q.ndim - 2 - 4
        # print 'LQ c', L.shape, Q.shape, num_io
        tens = np.tensordot(L,Q,axes=(-1-num_io,0))
        # print 'LQ c', tens.shape, axT_inv
    else:
        tens = np.tensordot(L,Q,axes=(-1,0))

    return tens.transpose(axT_inv)


def compress_peps_list_reg(pepx_list,connect_list,DMAX,direction=0):
    ''' compress list of peps connect via connect_list (str) '''

    new_list = pepx_list[:]

    if connect_list is None:  reorder = False
    else:                     reorder = True

    # print 'compress peps list', [m.shape for m in new_list]    

    ind = 0
    num_io = pepx_list[0].ndim-4

    if direction==1:
        new_list = new_list[::-1]
        if reorder:   connect_list = opposite_leg(connect_list)[::-1]

    s_list = []
    for ind in range(len(pepx_list)-1):

        if reorder: 
            iso1o = connect_list[ind]
            axT1, axT1_inv = get_io_axT('', iso1o, new_list[ind].ndim)
            pepx1 = new_list[ind].transpose(axT1)

            iso2i = opposite_leg(connect_list[ind])
            axT2, axT2_inv = get_io_axT(iso2i, '', new_list[ind+1].ndim)
            pepx2 = new_list[ind+1].transpose(axT2)
        else:
            if direction == 1:
                pepx1 = np.moveaxis(new_list[ind],  [0,-2],[-1,0])
                pepx2 = np.moveaxis(new_list[ind+1],[0,-2],[-1,0])
            else:   # direction == 0
                pepx1 = np.moveaxis(new_list[ind],  -1,-2)
                pepx2 = np.moveaxis(new_list[ind+1],-1,-2)
    

        block = np.tensordot(pepx1,pepx2,axes=(-1,0))
        u,s,vt,dwt = tf.svd(block,new_list[ind].ndim-1,DMAX)

        if reorder:
            new_list[ind]   = tf.dMult('MD',u,np.sqrt(s)) .transpose(axT1_inv)
            new_list[ind+1] = tf.dMult('DM',np.sqrt(s),vt).transpose(axT2_inv)
        else:
            if direction == 1:
                new_list[ind]   = np.moveaxis(tf.dMult('MD',u,np.sqrt(s)), [-1,0],[0,-2])
                new_list[ind+1] = np.moveaxis(tf.dMult('DM',np.sqrt(s),vt),[-1,0],[0,-2])
            else:
                new_list[ind]   = np.moveaxis(tf.dMult('MD',u,np.sqrt(s)), -2,-1)
                new_list[ind+1] = np.moveaxis(tf.dMult('DM',np.sqrt(s),vt),-2,-1)


    if direction==1:
        new_list = new_list[::-1]

    # print 'reduced compress', reorder, [m.shape for m in new_list]

    return new_list


def reduced_compress_peps_list(pepx_list,connect_list,DMAX,direction=1):
    ''' compress list of peps connect via connect_list (str) '''

    new_list = pepx_list[:]

    ind = 0
    num_io = pepx_list[0].ndim-4

    if direction==1:
        new_list = new_list[::-1]
        connect_list = opposite_leg(connect_list)[::-1]

    for ind in range(len(pepx_list)-1):

        iso1o = connect_list[ind]
        axT1, axT1_inv = get_io_axT('', iso1o, new_list[ind].ndim)
        pepx1 = new_list[ind].transpose(axT1)

        iso2i = opposite_leg(connect_list[ind])
        axT2, axT2_inv = get_io_axT(iso2i, '', new_list[ind+1].ndim)
        pepx2 = new_list[ind+1].transpose(axT2)

        Q1,R1 = tf.qr(pepx1,  4-len(iso1o) )
        L2,Q2 = tf.lq(pepx2,-(4-len(iso1o)))

        block = np.tensordot(R1,L2,axes=(-1,0))
        u,s,vt,dwt = tf.svd(block,R1.ndim-1,DMAX)

        new1 = np.tensordot(Q1,u,axes=(-1,0))
        new2 = np.tensordot(tf.dMult('DM',s,vt),Q2,axes=(-1,0))

        new_list[ind]   = new1.transpose(axT1_inv)
        new_list[ind+1] = new2.transpose(axT2_inv)

    if direction==1:
        new_list = new_list[::-1]

    return new_list


def reduced_compress_peps_list_reg(pepx_list,connect_list,DMAX,direction=1):
    ''' compress list of peps connect via connect_list (str) '''

    new_list = pepx_list[:]

    ind = 0
    num_io = pepx_list[0].ndim-4

    if direction==1:
        new_list = new_list[::-1]
        connect_list = opposite_leg(connect_list)[::-1]

    for ind in range(len(pepx_list)-1):

        iso1o = connect_list[ind]
        axT1, axT1_inv = get_io_axT('', iso1o, new_list[ind].ndim)
        pepx1 = new_list[ind].transpose(axT1)

        iso2i = opposite_leg(connect_list[ind])
        axT2, axT2_inv = get_io_axT(iso2i, '', new_list[ind+1].ndim)
        pepx2 = new_list[ind+1].transpose(axT2)

        Q1,R1 = tf.qr(pepx1,  4-len(iso1o) )
        L2,Q2 = tf.lq(pepx2,-(4-len(iso1o)))

        block = np.tensordot(R1,L2,axes=(-1,0))
        u,s,vt,dwt = tf.svd(block,R1.ndim-1,DMAX)

        # print Q1.shape, u.shape, R1.shape, axT1_inv

        new1 = np.tensordot(Q1,tf.dMult('MD',u,np.sqrt(s)),axes=(-1,0))
        new2 = np.tensordot(tf.dMult('DM',np.sqrt(s),vt),Q2,axes=(-1,0))

        new_list[ind]   = new1.transpose(axT1_inv)
        new_list[ind+1] = new2.transpose(axT2_inv)

    if direction==1:
        new_list = new_list[::-1]

    return new_list


def compress_peps_list(pepx_list,connect_list,DMAX,direction=1,num_io=1,regularize=True,canonicalize=False):
    ''' compress list of peps connect via connect_list (str)
        after compression, regularize
    '''
    
    return mpo_update(pepx_list,connect_list,None,DMAX=DMAX,num_io=num_io,direction=direction,regularize=regularize)


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


def mul_pepx_list(pepx_list,const):
    
    L = len(pepx_list)

    new_list = []
    for ind in range(L):
        new_list.append( pepx_list[ind]*(const**(1./L)) )

    return new_list


def norm_pepx_list(pepx_list,connect_list=None):
    ''' assumes surrounding env is identity '''

    if connect_list is not None:
        pepx_list = connect_pepx_list(pepx_list,connect_list)

    block = np.eye(pepx_list[0].shape[0])
    for ind in range(len(pepx_list)):
        ndim = pepx_list[ind].ndim
        # block = np.tensordot(block,pepx_list[ind],axes=(-1,0))
        block = np.einsum('ij,j...->i...',block,pepx_list[ind])
        block = np.tensordot(np.conj(pepx_list[ind]),block,axes=(range(ndim-1),range(ndim-1)))

    return np.sqrt(np.einsum('ii',block))


def canonicalize_list(pepx_list, direction=0, connect_list=None):

    if connect_list is not None:
        new_list, axTs = connect_pepx_list(pepx_list,connect_list)
    else:
        new_list = pepx_list[:]

    L = len(pepx_list)

    if direction == 0:
        for ind in range(0,L-1):
            q, r = tf.qr(new_list[ind],-1)
            # print new_list[ind].shape, q.shape, r.shape
            new_list[ind] = q
            new_list[ind+1] = np.tensordot(r,new_list[ind+1],axes=(-1,0))
            # block = np.tensordot(new_list[ind],new_list[ind+1],axes=(-1,0))
            # u,s,vt,dwt = tf.svd(block,block.ndim/2,-1)
            # new_list[ind]   = u
            # new_list[ind+1] = tf.dMult('DM',s,vt)
            # errs.append(dwt)
            # s_list.append(s)
    elif direction == 1:
        for ind in range(L-1,0,-1):
            l, q = tf.lq(new_list[ind],1)
            new_list[ind]   = q
            new_list[ind-1] = np.tensordot(new_list[ind-1],l,axes=(-1,0)) 
            # block = np.tensordot(new_list[ind-1],new_list[ind],axes=(-1,0))
            # u,s,vt,dwt = tf.svd(block,block.ndim/2,-1)
            # new_list[ind-1] = tf.dMult('MD',u,s)
            # new_list[ind]   = vt
            # errs.append(dwt)
            # s_list.append(s)

        # s_list = s_list[::-1]
        # errs = errs[::-1]
    else:
        pass
        # errs = np.nan


    if connect_list is not None:
        new_list = transpose_pepx_lits(new_list,axTs)
    
    return new_list
   


def check_canon_pepx_list(pepx_list,canon='L',connect_list=None):

    if connect_list is not None:
        pepx_list = connect_pepx_list(pepx_list,connect_list)

    
    if canon in [0,'l','L']:
        ndim = pepx_list[0].ndim
        tens = np.tensordot(np.conj(pepx_list[0]),pepx_list[0],axes=(range(ndim-1),range(ndim-1)))

        for ind in range(1,len(pepx_list)):
            ndim = pepx_list[ind].ndim
            tens = np.tensordot(tens,np.conj(pepx_list[ind]),axes=(0,0))
            tens = np.tensordot(tens,pepx_list[ind],axes=(range(ndim-1),range(ndim-1)))

        ndim = tens.shape[0]
        err = np.linalg.norm( tens - np.eye(ndim) )

    elif canon in [1,'r','R']:
        ndim = pepx_list[-1].ndim
        tens = np.tensordot(np.conj(pepx_list[-1]),pepx_list[-1],axes=(range(1,ndim),range(1,ndim)))

        for ind in range(len(pepx_list)-2,-1,-1):
            ndim = pepx_list[ind].ndim
            tens = np.tensordot(np.conj(pepx_list[ind]),tens,axes=(-1,0))
            tens = np.tensordot(tens,pepx_list[ind],axes=(range(1,ndim),range(1,ndim)))

        ndim = tens.shape[0]
        err = np.linalg.norm( tens - np.eye(ndim) )
        
    return err


def regularize_loop(sub_pepx, conn_list=None, ind0_list=None):

    new_pepx = sub_pepx.copy()

    assert(sub_pepx.shape==(2,2)), 'loop pepx needs to be 2x2 subpepx,not %dx%d'%sub_pepx.shape

    if conn_list is None:
        conn_list = ['oli','lir','iro','rol']
        ind0_list = [(0,1),(1,1),(1,0),(0,0)]
  
    for ind in range(len(conn_list)):
        xs,ys = get_conn_inds(conn_list[ind],ind0_list[ind])
        sub_list, axTs = get_sites(new_pepx,ind0_list[ind],conn_list[ind])
        
        if ind == len(conn_list)-1:
            sub_list,errs = compress_peps_list(sub_list,None,-1,ind%2,regularize=True)
        else:
            sub_list,errs = compress_peps_list(sub_list,None,-1,ind%2,regularize=False)

        new_pepx = set_sites(pepx,peps_list,ind0,connect_list,axT_invs)

    return new_pepx



# def reduced_compress_peps_list_reg2(pepx_list,connect_list,DMAX,direction=1):
#     ''' compress list of peps connect via connect_list (str)
#         after compression, regularize
#     '''
# 
#     new_list = reduced_compress_peps_list(pepx_list,connect_list,DMAX,direction=direction)
#     reg_list = reduced_compress_peps_list_reg(new_list,connect_list,-1,direction=(direction+1)%2)
# 
#     return reg_list


def regularize(pepx,idx):
    ''' pepx is (sub)pepx to be regularized
        idx is site to be regularized.
    '''

    L1,L2 = pepx.shape
    i1,i2 = idx

    print 'reg', idx, pepx.shape

    conns = []
    t_idx = []
    if idx[1] > 0:
        conns += ['l']
        t_idx += [(i1,i2-1)]
    if idx[0] > 0:
        conns += ['i']
        t_idx += [(i1-1,i2)]
    if idx[0] < L1-1:
        conns += ['o']
        t_idx += [(i1+1,i2)]
    if idx[1] < L2-1:
        conns += ['r']
        t_idx += [(i1,i2+1)]
 
    norms = [np.linalg.norm(pepx[i]) for i in t_idx]
    minxx = np.argmin( norms )

    # regularization of pepx[idx] with connected tens with min norm
    iso1o = conns[minxx]
    axT1, axT1_inv = get_io_axT('', iso1o, pepx[idx].ndim)
    pepx1 = pepx[idx].transpose(axT1)

    idx2  = t_idx[minxx]
    iso2i = opposite_leg(iso1o)
    axT2, axT2_inv = get_io_axT(iso2i, '', pepx[idx2].ndim)
    pepx2 = pepx[idx2].transpose(axT2)

    block = np.tensordot(pepx1,pepx2,axes=(-1,0))
    u,s,vt,dwt = tf.svd(block,pepx[idx].ndim-1,-1)

    pepx_ = copy(pepx)
    pepx_[idx]  = tf.dMult('MD',u,np.sqrt(s)). transpose(axT1_inv)
    pepx_[idx2] = tf.dMult('DM',np.sqrt(s),vt).transpose(axT2_inv)

    return pepx_
    
  
def regularize_all(pepx):
    ''' regularize all sites in pepx in a sweep fashion'''

    L1, L2 = pepx.shape

    norms = np.array([np.linalg.norm(m) for idx,m in np.ndenumerate(pepx)]).reshape((L1,L2))
    not_reg_enough = (np.max(norms)-np.min(norms) > 0.5)

    pepx_ = copy(pepx)

    while not_reg_enough:
        # print norms.shape, np.argmax(norms)
        x0,y0 = np.unravel_index(np.argmax(norms), norms.shape)

        if x0 == 0:   ix = 0
        else:         ix = 1

        if y0 == 0:   iy = 0
        else:         iy = 1

        sub_pepx = pepx_[x0-ix:x0+2,y0-iy:y0+2]
        reg_pepx = regularize(sub_pepx,(ix,iy))

        pepx_[x0-ix:x0+2,y0-iy:y0+2] = reg_pepx
        norms[x0-ix:x0+2,y0-iy:y0+2] = np.array([np.linalg.norm(m) for idx,m in \
                                                 np.ndenumerate(reg_pepx)]).reshape(reg_pepx.shape)

        not_reg_enough = (np.max(norms)-np.min(norms) > 10.)
        
       
    return pepx_


###########################################################
#### apply operators (trotter steps) to PEPX or PEPO ######
###########################################################

def split_singular_vals(pepx_list,connect_list,s_list,direction=0):

    ''' direction is canonicalization direction of pepx_list'''

    reg_list = []

    # print [m.shape for m in pepx_list]

    if connect_list is None:   # assume tensors already in correct a..(u)db order

        # split singular values
        ptens = pepx_list[0]
        if direction == 0:      ptens = tf.dMult('MD',ptens,np.sqrt(s_list[0]))
        elif direction == 1:    ptens = tf.dMult('MD',ptens,1./np.sqrt(s_list[0]))
        reg_list.append(ptens)

        for ind in range(1,len(pepx_list)-1):
            ptens = pepx_list[ind]
            if direction == 0:
                ptens = tf.dMult('DM',1./np.sqrt(s_list[ind-1]),ptens)
                ptens = tf.dMult('MD',ptens,np.sqrt(s_list[ind]))
            elif direction == 1:
                ptens = tf.dMult('DM',np.sqrt(s_list[ind-1]),ptens)
                ptens = tf.dMult('MD',ptens,1./np.sqrt(s_list[ind]))
            else:  
                raise ValueError('PEPX: split singular vals direction error, not '+str(direction))
            reg_list.append(ptens)

        ptens = pepx_list[-1]
        if direction == 0:      ptens = tf.dMult('DM',1./np.sqrt(s_list[-1]),ptens)
        elif direction == 1:    ptens = tf.dMult('DM',np.sqrt(s_list[-1]),ptens)
        reg_list.append(ptens)

    else:
        # split singular values
        isoR = connect_list[0]
        ptens = pepx_list[0]
        axT, axT_inv = get_io_axT('',isoR,ptens.ndim)
        ptens = ptens.transpose(axT)
        # print 'axT', axT, axT_inv, len(s_list[0]),connect_list
        if direction == 0:      ptens = tf.dMult('MD',ptens,np.sqrt(s_list[0]))
        elif direction == 1:    ptens = tf.dMult('MD',ptens,1./np.sqrt(s_list[0]))
        reg_list.append(ptens.transpose(axT_inv))

        for ind in range(1,len(pepx_list)-1):
            isoL = opposite_leg(isoR)
            isoR = connect_list[ind]
            ptens = pepx_list[ind]
            axT,axT_inv = get_io_axT(isoL,isoR,ptens.ndim)
            ptens = ptens.transpose(axT)
            if direction == 0:
                ptens = tf.dMult('DM',1./np.sqrt(s_list[ind-1]),ptens)
                ptens = tf.dMult('MD',ptens,np.sqrt(s_list[ind]))
            elif direction == 1:
                ptens = tf.dMult('DM',np.sqrt(s_list[ind-1]),ptens)
                ptens = tf.dMult('MD',ptens,1./np.sqrt(s_list[ind]))
            reg_list.append(ptens.transpose(axT_inv))

        isoL = opposite_leg(isoR)
        ptens = pepx_list[-1]
        axT,axT_inv = get_io_axT(isoL,'',ptens.ndim)
        ptens = ptens.transpose(axT)
        # print 'axT', axT, axT_inv, len(s_list[-1]),connect_list
        if direction == 0:      ptens = tf.dMult('DM',1./np.sqrt(s_list[-1]),ptens)
        elif direction == 1:    ptens = tf.dMult('DM',np.sqrt(s_list[-1]),ptens)
        reg_list.append(ptens.transpose(axT_inv))

        # print 'split sing vals', connect_list, isoL
        # print [m.shape for m in pepx_list]
        # print [m.shape for m in reg_list]

    return reg_list


def mpo_update(pepx_list, connect_list, mpo1, mpo2=None, DMAX=100, num_io=1, direction=0, regularize=True):
    ''' algorithm:  mpo1 is already decomposed... so just need to do contractions + compression
                    but maybe mpo representation of exp() is not great? (compression of mpo's not so well behaved?)
        contraction + compression:  qr + svd compression
        connect_list:  string of 'l','u','d','r' denoting path of mpo and pepx_list
    '''

    L = len(pepx_list)

    # num_io = pepx_list[0].ndim-4

    if connect_list is None:
        pepx_list = pepx_list[:]
    else:
        pepx_list, axTs = connect_pepx_list(pepx_list, connect_list)

    # print 'mpo update orig', [m.shape for m in pepx_list]

    new_list = []
    errs = []
    s_list = []

    for ind in range(0,L):
       
        tens = pepx_list[ind]

        # this way we can more easily integrate applying mpo to (Q)R gammas
        if num_io == 1:
            try:
                tens = np.einsum('LudR,l...dx->Ll...uRx',mpo1[ind],tens)
                tens = tf.reshape(tens,'ii,...,ii')
            except(TypeError):    # mpo1 is None
                pass
        elif num_io == 2:
            try:
                tens = np.einsum('LudR,l...dDx->Ll...uDRx',mpo1[ind],tens)
                tens = tf.reshape(tens,'ii,...,ii')
            except(TypeError):   # mpo1 is None
                pass
            try:
                tens = np.einsum('LUDR,l...dUx->lL...dDxR',mpo2[ind],tens)
                tens = tf.reshape(tens,'ii,...,ii')
            except(TypeError):   # mpo2 is None
                pass
        else:
            raise(NotImplementedError)

        # try:
        #     tens = np.einsum('LudR,labd...x->Llabu...Rx',mpo1[ind],tens) 
        #     tens = tf.reshape(tens,'ii,...,ii')
        # except(TypeError):   # mpo1 is None
        #     pass
        # try: 
        #     tens = np.einsum('LdDR,l...dx->lL...DxR',tens,mpo2[0])
        #     tens = tf.reshape(tens,'ii,...,ii')
        # except(TypeError):   # mpo2 is none
        #     pass

        new_list.append(tens)

    # print 'mpo update new', [m.shape for m in new_list]

    if direction == 0:
        new_list = canonicalize_list(new_list,direction=1)
        for ind in range(0,L-1):
            block = np.tensordot(new_list[ind],new_list[ind+1],axes=(-1,0))
            u,s,vt,dwt = tf.svd(block,block.ndim/2,DMAX)
            new_list[ind]   = u
            new_list[ind+1] = tf.dMult('DM',s,vt)
            errs.append(dwt)
            s_list.append(s)
    elif direction == 1:
        new_list = canonicalize_list(new_list,direction=0)
        for ind in range(L-1,0,-1):
            block = np.tensordot(new_list[ind-1],new_list[ind],axes=(-1,0))
            u,s,vt,dwt = tf.svd(block,block.ndim/2,DMAX)
            new_list[ind-1] = tf.dMult('MD',u,s)
            new_list[ind]   = vt
            errs.append(dwt)
            s_list.append(s)

        s_list = s_list[::-1]
        errs = errs[::-1]

    else:
        errs = np.nan

    # print '2', [m.shape for m in new_list]

    if regularize:
        new_list = split_singular_vals(new_list,None,s_list,direction=direction)
        # connect_list None: tens dims order as is
 

    # reorder axes to liord if originally in that order
    if connect_list is not None:
        for ind in range(0,L):
            new_list[ind] = new_list[ind].transpose(axTs[ind])

    return new_list, errs


# def mpo_update(pepx_list, connect_list, mpo1, mpo2=None, DMAX=100, direction=0, regularize=True):
#     ''' algorithm:  mpo1 is already decomposed... so just need to do contractions + compression
#                     but maybe mpo representation of exp() is not great? (compression of mpo's not so well behaved?)
#         contraction + compression:  qr + svd compression
#         connect_list:  string of 'l','u','d','r' denoting path of mpo and pepx_list
#     '''
# 
#     L = len(pepx_list)
#     # print 'mpo update len', L, connect_list
#     new_list = []
#     
#     errs = []
#     s_list = []
# 
#     num_io = pepx_list[0].ndim-4
# 
#     axTs = []
#     for ind in range(0,L):
#        
#         try:   isoL = opposite_leg(isoR)
#         except(NameError):    isoL = opposite_leg(connect_list[0])
#        
#         try:   isoR = connect_list[ind]
#         except(IndexError):   isoR = opposite_leg(isoL)
# 
#         axT,axT_inv = get_io_axT(isoL,isoR,pepx_list[ind].ndim)
#         tens = pepx_list[ind].transpose(axT)
#         try:
#             tens = np.einsum('LudR,labd...x->Llabu...Rx',mpo1[ind],tens) 
#             tens = tf.reshape(tens,'ii,...,ii')
#         except(TypeError):   # mpo1 is None
#             pass
#         try: 
#             tens = np.einsum('LdDR,l...dx->lL...DxR',tens,mpo2[0])
#             tens = tf.reshape(tens,'ii,...,ii')
#         except(TypeError):   # mpo2 is none
#             pass
#         new_list.append(tens)
# 
#         axTs.append(axT_inv)
# 
#     # print '1', [m.shape for m in new_list]
#     # print axTs
#     
#     if direction == 0:
#         for ind in range(0,L-1):
#             block = np.tensordot(new_list[ind],new_list[ind+1],axes=(-1,0))
#             u,s,vt,dwt = tf.svd(block,block.ndim/2,DMAX)
#             new_list[ind]   = u
#             new_list[ind+1] = tf.dMult('DM',s,vt)
#             errs.append(dwt)
#             s_list.append(s)
#     elif direction == 1:
#         for ind in range(L-1,0,-1):
#             block = np.tensordot(new_list[ind-1],new_list[ind],axes=(-1,0))
#             u,s,vt,dwt = tf.svd(block,block.ndim/2,DMAX)
#             new_list[ind-1] = tf.dMult('MD',u,s)
#             new_list[ind]   = vt
#             errs.append(dwt)
#             s_list.append(s)
# 
#         s_list = s_list[::-1]
# 
#     # print '2', [m.shape for m in new_list]
# 
#     if regularize:
#         new_list = split_singular_vals(new_list,None,s_list,direction=direction)
#         # connect_list None: tens dims order as is
#  
# 
#     # reorder axes to liord
#     for ind in range(0,L):
#         new_list[ind] = new_list[ind].transpose(axTs[ind])
# 
#     # print '3', [m.shape for m in new_list]
# 
#     return new_list, errs



def reduced_mpo_update(pepx_list, connect_list, mpo1, mpo2=None, DMAX=100, direction=0, regularize=True):
    ''' algorithm:  mpo1 is already decomposed... so just need to do contractions + compression
                    but maybe mpo representation of exp() is not great? (compression of mpo's not so well behaved?)
        contraction + compression:  qr + svd compression
        connect_list:  string of 'l','u','d','r' denoting path of mpo and pepx_list
    '''

    new_list = pepx_list[:]
    errs = []
    s_list = []

    num_io = pepx_list[0].ndim-4

    # print 'reduced mpo update', connect_list
    
    if connect_list is not None:
    
        ind = 0
        iso1 = connect_list[ind]
    
        Q1, R1, axT_inv1 = QR_factor(new_list[ind],iso1)
        R1 = np.einsum('AudB,ad...b->Aau...Bb',mpo1[ind],R1)
        R1 = tf.reshape(R1,'ii,...,ii')
        try:
            R1 = np.einsum('audb,AdDB->aAuDbB',R1,mpo2[ind])
            R1 = tf.reshape(R1,'ii,...,ii')
        except:   pass
        
        ind = 1
        while ind < len(pepx_list):
    
            iso2 = opposite_leg(iso1)   # which leg to contract based on direction of previous tens position
           
            L2, Q2, axT_inv2 = LQ_factor(new_list[ind],iso2)
            L2 = np.einsum('BudC,bd...c->Bbd...Cc',mpo1[ind],L2)
            L2 = tf.reshape(L2,'ii,...,ii')
            try:
                L2 = np.einsum('budc,BdDC->bBuDcC',L2,mpo2[ind])
                L2 = tf.reshape(L2,'ii,...,ii')
            except:  pass
    
            # svd compression of R1*L2
            X12 = np.tensordot(R1,L2,axes=(-1,0))
            u,s,vt,dwt = tf.svd(X12,R1.ndim-1,DMAX)
            R1_ = u
            L2_ = tf.dMult('DM',s,vt)
            # R1_ = tf.dMult('MD',u, s)
            # L2_ = vt
            errs.append(dwt)
            s_list.append(s)
    
            # next step:  LQ -> QR (in new direction)
            new_list[ind-1] = QR_contract(Q1,R1_,axT_inv1)
    
            pepx1 = LQ_contract(L2_,Q2,axT_inv2)
            try:
                iso1  = connect_list[ind] 
                Q1, R1, axT_inv1 = QR_factor(pepx1,iso1)
            except(IndexError):   # last element in list
                pass
    
            ind += 1
    
        new_list[-1] = LQ_contract(L2_,Q2,axT_inv2)
    
        if regularize:
            new_list = split_singular_vals(new_list,connect_list,s_list,direction=0)

    else:

        if mpo2 is not None:  raise(NotImplementedError)

        # as of now, only works for 2x1 trotter steps

        # print 'reduced mpo update', [m.shape for m in pepx_list]
 
        R1 = np.moveaxis(pepx_list[0],-1,-2)
        R1 = np.einsum('AudB,a...db->Aa...uBb',mpo1[0],R1)
        R1 = tf.reshape(R1,'ii,...,ii') 
        # try:
        #     R1 = np.einsum('audb,A...B->aAbBuD',R1,mpo2[0])
        #     R1 = tf.reshape(R1,'ii,...,ii')
        # except:   pass

        ind = 1
        while ind < len(pepx_list):
    
            L2 = np.moveaxis(new_list[ind],-1,-2)
            L2 = np.einsum('BudC,bd...c->Bbu...Cc',mpo1[ind],L2)
            L2 = tf.reshape(L2,'ii,...,ii') 

            # try:
            #     L2 = np.einsum('budc,BdDC->bBuDcC',L2,mpo2[ind])
            #     L2 = tf.reshape(L2,'ii,...,ii')
            # except:  pass
    
            # svd compression of R1*L2
            X12 = np.tensordot(R1,L2,axes=(-1,0))
            u,s,vt,dwt = tf.svd(X12,R1.ndim-1,DMAX)
            R1_ = u
            L2_ = tf.dMult('DM',s,vt)
            # R1_ = tf.dMult('MD',u, s)
            # L2_ = vt
            errs.append(dwt)
            s_list.append(s)
    
            # next step:  LQ -> QR (in new direction)
            new_list[ind-1] = np.moveaxis(R1_,-2,-1)
            R1_ = L2_

            ind += 1
    
        new_list[-1] = np.moveaxis(L2_,-2,-1)
 
    # print 'reduced mpo new list', [m.shape for m in new_list]

    return new_list, errs


def block_update(pepx_list, connect_list, block1, block2=None, DMAX=10, canon=0):
    ''' apply trotter operator (block) to pepx_list  '''

    new_block = block1.copy()

    reorder = (connect_list is not None)
    if reorder:   pepxT_list, axT_invs = connect_pepx_list(pepx_list,connect_list,side='R')
    else:         pepxT_list = peps_list[:]

    # block * peps
    if np.all([ptens.ndim == 5 for ptens in pepx_list]):   # peps

        peps1 = pepxT_list[0]
    
        new_block = np.einsum('AUD...,abcDe->AabcUe...',block1,peps1)
        new_block = tf.reshape(new_block,'ii,...')
    
        ind = 1
        while ind < len(peps_list):
        
            peps1 = pepxT_list[ind]
        
            p1 = 1 + ind*3 + 1
            p2 = p1 + 2
            p3 = p2 + 3 + (L-ind-1)*2 + 1
        
            indb = range(p1) + range(p2,p2+2) + range(p2+3,p3)
            inds = range(p1-1,p2) + range(p2+1,p2+3)
    
            new_block = np.einsum(new_block,indb,peps1,inds)
        
            ind += 1
        
        new_block = tf.reshape(new_block,'...,ii')
    
        # # svd
        # gam_list, lam_list, errs = tf.decompose_block_GL(new_block,L,DMAX,svd_str=4)

        # # reorder indices
        # if reorder:     gam_list = PEPX.transpose_pepx_list(gam_list,axT_invs)
        # else:           gam_list = gam_list
    
        new_list, s_list = tf.decompose_block(new_block,L,direction,DMAX,svd_str=4,return_s=True)    # left canonical
       
        # reorder indices
        if reorder:     block_list = transpose_pepx_list(new_list,axT_invs)
        else:           block_list = new_list

    # block * pepo
    elif np.all([ptens.ndim == 6 for ptens in pepx_list]):   # pepo

        if block2 is None: 
            pepo1 = pepxT_list[0]
    
            new_block = np.einsum('AUD...,abceDd->AabcUde...',block1,pepo1)
            new_block = tf.reshape(new_block,'ii,...')
        
            ind = 1
            while ind < len(pepo_list):

                pepo1 = pepxT_list[ind]
            
                p1 = 1 + ind*4 + 1
                p2 = p1 + 2
                p3 = p2 + 4 + (L-ind-1)*2 + 1
        
                indb = range(p1) + range(p2,p2+2) + range(p2+4,p3)
                inds = range(p1-1,p2) + range(p2+1,p2+3)
                new_block = np.einsum(new_block,indb,pepo1,inds)
        
                ind += 1
        
            new_block = tf.reshape(new_block,'...,ii')

        else:    # include tensor dot block2
            pepo1 = pepxT_list[0]

            indm1_ = [[0] + [5,6]] + [[i,i+1] for i in range(10,10+(L-1)*4,4)] + [[10+(L-1)*4]]
            inds   = [1,3,4,6,7,9]
            indm2_ = [[2] + [7,8]] + [[i,i+1] for i in range(12,10+(L-1)*4,4)] + [[10+(L-1)*4+1]]

            # flatten lists
            indm1 = [x for y in indm1_ for x in y]
            indm2 = [x for y in indm2_ for x in y]

            new_block = np.einsum(block1,indm1,pepo1,inds,block2,indm2)
            new_block = tf.reshape(new_block,'iii,...')

            ind = 1
            while ind < len(pepo_list):

                pepo1 = pepxT_list[ind]
            
                p1 = 1 + ind*4 + 1
                p2 = p1 + 2
                p3 = p2 + 5 + (L-ind-1)*4 + 2
        
                indb = range(p1) + range(p2,p2+2) + range(p2+2,p2+4) + range(p2+5,p3)
                inds = range(p1-1,p2) + range(p2+1,p2+3) + [p2+4]

                new_block = np.einsum(new_block,indb,pepo1,inds)

                ind += 1
        
            new_block = tf.reshape(new_block,'...,iii')

    
        # svd
        new_list, s_list = tf.decompose_block(new_block,L,direction,DMAX,svd_str=4,return_s=True)    # left canonical

       
        # reorder indices
        if reorder:     block_list = transpose_pepx_list(new_list,axT_invs)
        else:           block_list = new_list
    

    else:
        raise(TypeError), 'PEPX: pepx should be either MPS or MPO'

    return block_list
    


def get_qr_chain(pepx_list, connect_list):
    '''
    does qr on pepx_list --> R tens:  dl x dQ (contract with Q) x ( d x d) x dr
    connect_list = str denoting connection of tens in pepx_list
    returns lists of Q's, R's, and axT_inv
    '''

    L = len(pepx_list)
    Q_list = []
    R_list = []
    axT_inv_list = []

    for ind in range(L):
        
        try:  iso1i = opposite_leg(iso1o)    # ind = 0 case
        except(NameError):  iso1i = ''
        try:  iso1o = connect_list[ind]      # ind = L case
        except(IndexError):  iso1o = ''

        # axT1, axT1_inv = get_io_axT('',iso1o, pepx_list[ind].ndim)
        # pepo1 = pepx_list[ind].transpose(axT1)
        Q,r,axT_inv = QR_factor(pepx_list[ind],iso1i+iso1o)   # Q = tens_shape x dQ, R = dQ x tens_shape
        if   ind == 0:       # dQ x (d x d) x dr -> 1 x dQ x  d (x d) x dr
            R = r.reshape( (1,) + r.shape )
        elif ind == L-1:     # dQ x (d x d) x dr -> dr x dQ x (d x d) x 1
            r = np.moveaxis(r,[-1],[0])
            R = r.reshape( r.shape + (-1,) )
        else:                # dQ x (d x d) x dl x dr -> dl x dQ x d (x d) x dr
            R = np.moveaxis( r, [-2], [0] )

        Q_list.append( Q )
        R_list.append( R )
        axT_inv_list.append( axT_inv )
    
    return Q_list, R_list, axT_inv_list


def contract_qr_chain(Q_list, R_list, axT_inv_list):
    ''' contracts Q, R's to get back pepx_list
    '''
    L = len(pepx_list)
    tens_list = []

    for ind in range(L):
        if ind == 0:
            R_ = R_list[ind]
            R  = R_.reshape( R_[1:] )
        elif ind == L-1:
            R_ = R_list[ind]
            R  = R_.reshape( R_[:-1] )
            R  = np.moveaxis(R,[0],[-1])
        else:
            R  = np.moveaxis(R_list[ind],[0],[-2])

        tens = QR_contract(Q_list[ind],R,axT_inv_list[ind])
        tens_list.append(tens)

    return tens_list


##########################################################
############ loop contraction methods #############
##########################################################

def update_loop(sub_pepx, mpo1=None, ind0=None, connect_list=None, DMAX=10, mpo2=None):

    assert(sub_pepx.shape == (2,2)), 'sub pepx needs to be 2x2 loop'

    pepx_ = get_pepx(sub_pepx)

    if mpo1 is not None: 
       pepx_list, axTs = get_sites(pepx_,ind0,connect_list)
       new_list, errs =  mpo_update(pepx_list, None, mpo1, mpo2=mpo2, DMAX=-1, regularize=False)
       new_pepx = set_sites(pepx_,new_list,ind0,connect_list,axTs)
    else:
       new_pepx = pepx_

    ### contract loop ###
    if new_pepx[0,0].ndim == 5:
        loop = np.einsum('liord,rjpse->liodjpse',new_pepx[0,0],new_pepx[0,1])
        loop = np.einsum('liodjpse,LoORD->lidjpseLORD',loop,new_pepx[1,0])
        loop = np.einsum('lidjpseLORD,RpPSE->lidjseLODPSE',loop,new_pepx[1,1])

        u,s,vt,dwt = tf.svd(loop,loop.ndim/2,2*DMAX)    # (0,0)+(0,1) , (1,0)+(1,1)
  
        two_site1 = tf.dMult('MD',u,np.sqrt(s))
        two_site2 = tf.dMult('DM',np.sqrt(s),vt)

        b1dim = len(s)
        b2dim = int(b1dim/2)

        iden = np.eye(b1dim)[:,:2*b2dim]
        isometry = iden.reshape(b1dim,b2dim,b2dim)

        two_site1 = np.einsum('lidjseX,Xop->liodjpse',two_site1,isometry)
        two_site2 = np.einsum('XLODPSE,XIJ->LIODJPSE',two_site2,isometry)

        u1,s1,vt1,dwt = tf.svd(two_site1,two_site1.ndim/2,DMAX)
        u2,s2,vt2,dwt = tf.svd(two_site2,two_site2.ndim/2,DMAX)

        new_pepx[0,0] = tf.dMult('MD',u1,np.sqrt(s1)).transpose(0,1,2,4,3)
        new_pepx[0,1] = tf.dMult('DM',np.sqrt(s1),vt1)
        new_pepx[1,0] = tf.dMult('MD',u2,np.sqrt(s2)).transpose(0,1,2,4,3)
        new_pepx[1,1] = tf.dMult('DM',np.sqrt(s2),vt2)

    else:
        raise(NotImplementedError)
     
    return new_pepx   


#############################################################
################## measurement methods ######################
#############################################################



def meas_corrs(pepx,idx,obs=None):

    corrs = np.ndarray(pepx.shape)

    for idx2 in np.ndindex(pepx.shape):

        if idx2 == idx:
            corrs[idx] = 1.
            continue

        corrs[idx2] = meas_corr(pepx,idx,idx2,obs)
    
    return corrs
       
  
