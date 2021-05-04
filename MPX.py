import numpy as np
import scipy.linalg as LA
import time
import tens_fcts as tf



class MPX(np.ndarray):
    """
    Lightweight MPS/MPO class
    """

    def __new__(cls,mparray,phys_bonds=None):
        """
        Parameters
        ----------
        mparray [array_like]:  1-D list/ndarray of tensors

        """
 
        if isinstance(mparray,np.ndarray) or isinstance(mparray,cls):
            mpx = mparray.view(cls)
        else:
            ## mpx is a list
            mpx = np.empty(len(mparray),dtype=object)
            mpx[:] = mparray[:]
            mpx = mpx.view(cls)
            # mpx = np.asarray(mparray,dtype=np.object).view(cls)

        if mpx.ndim > 1:   raise TypeError, 'mparray must be 1D not %d' %mpx.ndim
  
        if phys_bonds is None:
            mpx.phys_bonds = np.empty(mpx.shape,dtype=tuple)
            for idx, x in np.ndenumerate(mpx):
                mpx.phys_bonds[idx] = x.shape[1:-1]
        else:
            try:
                if phys_bonds.dtyep == 'O':
                    mpx.phys_bonds = phys_bonds[:]
                else:
                    raise (AttributeError)
            except(AttributeError):     # is a list / not an an array of object type
                mpx.phys_bonds = np.empty(len(phys_bonds),dtype=tuple)
                for i in range(len(phys_bonds)):
                    try:
                        mpx.phys_bonds[i] = tuple(phys_bonds[i])
                    except (TypeError):    # d is an int
                        mpx.phys_bonds[i] = (phys_bonds[i],)

                # mpx.phys_bonds = np.asarray([d for d in phys_bonds],dtype=object)
                # mpx.phys_bonds = np.asarray(phys_bonds,dtype=np.object)

        return mpx


    def __array_finalize__(self,mpx):
        if mpx is None:  return
        self.phys_bonds = getattr(mpx, 'phys_bonds',None)  


    def __getitem__(self,item):
        sliced_mpx = super(MPX,self).__getitem__(item)
        try:
            sliced_mpx.phys_bonds = self.phys_bonds.__getitem__(item)
        except(AttributeError):   
            # np.ndarray has no attribute phys_bonds.  occurs when trying to print? i guess it's called recursively
            pass
        return sliced_mpx

    def __setitem__(self,item,y):
        super(MPX,self).__setitem__(item,y)
        try:                      # y is also an MPX object
            self.phys_bonds.__setitem__(item,y.phys_bonds)
        except(AttributeError):   # y is just an ndarray

            # print self.phys_bonds
            if isinstance(item,int):
                temp_bonds = y.shape[1:-1]
            else:
                ## item is an iterable
                temp_bonds = np.empty(y.shape,dtype=object)
                for idx, yy in np.ndenumerate(y):
                    temp_bonds[idx] = yy.shape[1:-1]

            self.phys_bonds.__setitem__(item,temp_bonds)


    def __add__(mpx1, mpx2):
        return add(mpx1, mpx2)
    
    def __sub__(mpx1, mpx2):
        # return add(mpx1, mpx2*-1)
        return add(mpx1, mul(-1,mpx2))

    def __mul__(mpx, alpha):
        return mul(alpha, mpx)

    def __neg__(mpx):
        return mpx.mul(-1)
    
    # def __array_ufunc__(mpx, ufunc, method, *inputs, **kwargs):
    #     return mpx.__array_ufunc__(ufunc, method, inputs, kwargs)

    def dot(mpx1, mpx2):
        return dot(mpx2,mpx1)

    def dot_compress(mpx1,mpx2,D,direction=0):
        return dot_compress(mpx1,mpx2,D,direction)

    def norm(mpx):
        return norm(mpx)

    def transposeUD(mpx):
        return transposeUD(mpx)

    def outer(mpx1,mpx2):
        return outer(mpx1,mpx2)

    def getSites(mpx1,ind0=0,ns=None):
        if ns is None:  ns = len(mpx1)
        return getSites(mpx1,ind0,ns)

    def copy(mpx1):
        return copy(mpx1)


####################################
#########   MPX fcts   #############
####################################


##### functions to create MPX ######

def create(dp, D=None, fn=np.zeros, split=None):
    """
    Create random MPX object as ndarray of ndarrays

    Parameters
    ----------
    dp : list of ints or list of 2-tuples
      Specifies physical dimensions of MPS or MPO. 
    D : int, maximum bond dimension

    Returns
    -------
    mpx : MPX object, ndarray of ndarrays
       MPS or MPO
    """

    L   = len(dp)
    mpx = np.empty(L, dtype=np.object)

    # dp_ = [tuple(np.ravel(d)) for d in dp]    # to allow for nested bonds
    _dp = [np.prod(d) for d in dp]
        
    # calculate right bond dims of each tensor
    dim_rs = np.append(1, np.append( calc_dim(_dp,D), 1) )

    # fill in MPX with random arrays of the correct shape
    for i in range(0, L):
        mpx[i] = fn((dim_rs[i], _dp[i], dim_rs[i+1]))

    try:   # if dp was list of tuples, put back into original shape
        for i in range(L):
            mpx[i] = np.reshape(mpx[i], (mpx[i].shape[0],)+dp[i]+(mpx[i].shape[-1],))
    except(TypeError):  
        pass
     
    return MPX(mpx, phys_bonds=dp)

def empty(dp, D = None):
    return create(dp, D, fn=np.empty)

def zeros(dp, D = None):
    return create(dp, D, fn=np.zeros)

def ones(dp,D = None):
    return create(dp, D, fn=np.ones)

def rand(dp, D = None, seed = None):
    if seed is not None:
        np.random.seed(seed)
    return create(dp, D, fn=np.random.random)

  
def calc_dim(dp,D=None):
    """
    calculates bond dimension of 1-D MPX given physical bond dimensions

    Parameters
    -----------
    dp:   list/array of integers specifying the dimension of the physical bonds at each site
    D:    maximum bond dimension

    Returns
    -------
    list of the bond dimensions of the right leg to be used in generating the MPS

    """

    dimR = np.cumprod(dp)   # for very large systems, cumprod will go to neg vals
    dimL = np.cumprod(dp[::-1])[::-1]

    dimR[np.where(dimR <=0)[0]] = 1e15
    dimL[np.where(dimL <=0)[0]] = 1e15
    
    dimMin = np.minimum(dimR[:-1],dimL[1:])
    if D is not None:
        dimMin = np.minimum(dimMin,[D]*(len(dp)-1))

    return dimMin


def product_state(dp, occ):
    """
    generate MPS produuct state

    Parameters
    ----------
    dp:   list/array of integers specifying dimension of physical bonds at each site
    occ:  occupancy vector (len L), numbers specifying physical bond index occupied

    Returns
    -------
    returns product state mps according to occ
    """
    L = len(dp)
    mps = zeros(dp, 1)
    for i in range(L):
        mps[i][0, occ[i], 0] = 1.

    return mps


def product_mpo(dp, occ):
    """
    generate MPS produuct state

    Parameters
    ----------
    dp:   list/array of integers specifying dimension of physical bonds at each site
    occ:  occupancy vector (len L), numbers specifying physical bond index occupied

    Returns
    -------
    returns product state mps according to occ
    """
    L = len(dp)
    mpo = zeros(dp, 1)
    for i in range(L):
        mpo[i][0, occ[i][0], occ[i][1], 0] = 1.

    return mpo


def eye(dp):
    """
    generate identity MPO
    dp:  list of physical bonds (not tuples) at each site
    """
    L = len(dp)

    sites = []
    for d in dp:
        sites.append(np.eye(d).reshape(1,d,d,1))

    return MPX(sites)

    
def copy(mpx):
    # deeper copy than mpx.copy, as it copies the ndarrays in mpx
    sites = [m.copy() for m in mpx]
    return MPX(sites,mpx.phys_bonds)


#############################
####### other fcts? #########
#############################

def element(mpx, occ):
    """
    returns value of in mpx for given occupation vector
    """
    mats = [None] * len(mpx)
    try: # mpx is an mpo
        if len(occ[0]) == 2:
            for i, m in enumerate(mpx):
                mats[i] = m[i][:,occ[i][0],occ[i][1],:]
    except:
        for i, m in enumerate(mpx):
            mats[i] = m[:,occ[i],:]
        
    return np.asscalar(reduce(np.dot, mats))


def asfull(mpx):
    dp = tuple(m.shape[1] for m in mpx)

    n = np.prod(dp)
    dtype = mpx[0].dtype
    if mpx[0].ndim == 4: # mpx is an mpo
        dense = np.zeros([n, n], dtype=dtype)
        for occi in np.ndindex(dp):
            i = np.ravel_multi_index(occi, dp)
            for occj in np.ndindex(dp):
                j = np.ravel_multi_index(occj, dp)
                dense[i, j] = element(mpx, zip(occi, occj))
    else:
        assert mpx[0].ndim == 3 # mpx is an mps        
        dense = np.zeros([n], dtype=dtype)
        for occi in np.ndindex(dp):
            i = np.ravel_multi_index(occi, dp)
            dense[i] = element(mpx, occi)
    return dense


def check_canon(mpx,canon=0,ind0=None):

    if canon == 0:  # left

        stop_calc = False
        if ind0 is None:  ind0 = len(mpx)

        xx = 0
        for LL in mpx[:ind0]:
            LL_ = np.tensordot( np.conj(LL), LL, axes=(range(LL.ndim-1),range(LL.ndim-1)) )
            # print 'LL',np.linalg.norm(LL_-np.eye(LL_.shape[0]))
            if not np.allclose(LL_, np.eye(LL_.shape[0]),atol=1.0e-8*np.prod(LL_.shape)):
                print 'MPX not left canonical', xx, 'DMAX?', np.max([m.shape[0] for m in mpx])
                print np.linalg.norm(LL_ - np.eye(LL_.shape[0]))
                stop_calc = True
                # print np.abs( LL_ - np.eye(LL_.shape[0]) )
            xx += 1

        # ## norm
        # xx = 0
        # for LL in mpx:
        #     if xx == 0:  LL_left = LL
        #     else:        LL_left = np.einsum('ij,j...->i...',LL_, LL)
        #     LL_ = np.tensordot( np.conj(LL), LL_left, axes=(range(LL.ndim-1),range(LL.ndim-1)) )
 
        # print 'check canon norm', np.einsum('ii->',LL_)
    
    elif canon == 1:   # right

        stop_calc = False
        if ind0 is None:  ind0 = 0

        xx = 0
        for RR in mpx[ind0:]:
            RR_ = np.tensordot( np.conj(RR), RR, axes=(range(1,RR.ndim),range(1,RR.ndim)) )
            # print 'RR', np.linalg.norm(RR_-np.eye(RR_.shape[0]))
            if not np.allclose(RR_, np.eye(RR_.shape[0]),atol=1.0e-8*np.prod(RR_.shape)):
                print 'MPX not right canonical',xx, 'DMAX?', np.max([m.shape[0] for m in mpx])
                print np.linlg.norm(RR_ - np.eye(RR_.shape[0]))
                stop_calc = True
                # print np.abs( RR_ - np.eye(RR_.shape[0]))
            xx += 1

    if stop_calc:
        raise RuntimeError('not canonicalized')


##############################
#### arithemtic functions ####
##############################

def vdot(mps1, mps2, direction=0, as_array=False):
    """
    vdot of two MPS, returns scalar if as_array is False
    if as_arry is true, returns vdot as 'lLrR'

    cf. np.vdot
    """
    # try:
    #     return mps_dot(np.conj(mps1), mps2, direction)
    # except(ValueError):
    ovlp = mps_dot(np.conj(flatten(mps1)),flatten(mps2))

    if as_array:  return ovlp
    else:         return np.squeeze(ovlp)


def mps_dot(mps1, mps2, direction=0):
    """
    dot of two MPS, returns scalar
    """
    L = len(mps1)
    assert len(mps2) == L
    assert direction in (0, 1)
    
    # mpsdot = [ np.einsum('ldr,LdR->lLrR',mps1[i],mps2[i]) for i in range(L) ]
    # 
    # if direction == 0:
    #     E = mpsdot[0]
    #     for i in range(1,L):
    #         E = np.tensordot(E, mpsdot[i], axes=([-2,-1],[0,1]))
    # else:
    #     E = mpsdot[-1]
    #     for i in range(L-2,-1,-1):
    #         E = np.tensordot(mpsdot[i], E, axes=([-2,-1],[0,1]))

    if direction == 0:
        # E = np.einsum('lnr, LnR -> lLrR', mps1[0], mps2[0])
        E = np.tensordot(mps1[0],mps2[0],axes=(1,1))
        E = E.transpose(0,2,1,3)
        for i in range(1, L):
            # contract with bra
            # E = np.einsum('lLrR, rns -> lLRns', E, mps1[i])
            E = np.tensordot(E,mps1[i],axes=(2,0))   # lLRns
            # contract with ket
            # E = np.einsum('lLRns, RnS -> lLsS', E, mps2[i])
            E = np.tensordot(E,mps2[i],axes=([2,3],[0,1]))
            # E  = E2.copy()
        
        # if not np.allclose( np.einsum('iiab->ab',E) , np.eye(E.shape[-1]) ):
        #     print 'mps dot not L canonical'
        #     print E
 
    else:
        # E = np.einsum('lnr, LnR -> lLrR', mps1[-1], mps2[-1]) 
        E = np.tensordot(mps1[-1],mps2[-1],axes=(1,1))  #lrLR
        E = E.transpose(0,2,1,3)  # lLrR
        for i in range(L-2, -1, -1):
            # contract with bra
            # E = np.einsum('mnl,lLrR->mnLrR', mps1[i], E)
            E = np.tensordot(mps1[i],E,axes=(2,0))   # mnLrR
            # contract with ket
            # E = np.einsum('MnL,mnLrR->mMrR', mps2[i], E)
            E = np.tensordot(mps2[i],E,axes=([1,2],[1,2]))  # MmrR
            E = E.transpose(1,0,2,3)
            # E = E2.copy()

        # if not np.allclose( np.einsum('abii->ab',E) , np.eye(E.shape[0]) ):
        #     print 'mps dot not R canonical'
        #     print E
 

    # print 'mps dot', type(mps1[0]), mps1[0].dtype, type(mps2[0]), mps2[0].dtype
    # m1_s = mps1.getSites()
    # m2_s = mps2.getSites()
    # ovlp_test = np.tensordot( m1_s, m2_s, axes=(range(1,m1_s.ndim-1),range(1,m2_s.ndim-1))).transpose(0,2,1,3)
    # print E.shape, ovlp_test.shape
    # err = np.linalg.norm(E-ovlp_test)
    # print 'mps hand', err, np.einsum('llrr->',ovlp_test), np.einsum('llrr->',E), np.sqrt(np.einsum('llrr->',E))

    # if err > 1.0e-10: 
    #     print m1_s.shape, m2_s.shape, range(1,m1_s.ndim-1)

    #     # mpsdot = [ np.einsum('ldr,LdR->lLrR',mps1[i],mps2[i]) for i in range(L) ]
    #     mpsdot = [ np.tensordot(mps1[i],mps2[i],axes=(1,1)).transpose(0,2,1,3) for i in range(L) ]
    #     
    #     if direction == 0:
    #         Ey = mpsdot[0]
    #         for i in range(1,L):
    #             Ey = np.tensordot(Ey, mpsdot[i], axes=([-2,-1],[0,1]))
    #     else:
    #         Ey = mpsdot[-1]
    #         for i in range(L-2,-1,-1):
    #             Ey = np.tensordot(mpsdot[i], Ey, axes=([-2,-1],[0,1]))

    #     err  = np.linalg.norm(Ey-ovlp_test)
    #     err2 = np.linalg.norm(Ey-E)
    #     print 'mps hand exit', err, err2

    #     if direction == 0:
    #         Ex = np.einsum('lnr, LnR -> lLrR', mps1[0], mps2[0])
    #         for i in range(1, L):
    #             # contract with bra
    #             E1 = np.tensordot(Ex, mps1[i], axes=(-2,0))
    #             # contract with ket
    #             E2 = np.tensordot(E1, mps2[i], axes=([-3,-2],[0,1]))
    #             Ex = E2.copy()
    #     else:
    #         Ex = np.einsum('lnr, LnR -> lLrR', mps1[-1], mps2[-1]) 
    #         for i in range(L-2, -1, -1):
    #             # contract with bra
    #             E1 = np.tensordot(mps2[i],Ex,axes=(-1,1))
    #             # contract with ket
    #             E2 = np.tensordot(mps1[i],E1,axes=([-2,-1],[1,2]))
    #             Ex = E2.copy()

    #     print 'mps hand exit', np.linalg.norm(Ex-E), np.linalg.norm(Ex-Ey), np.linalg.norm(Ex-ovlp_test)

    #     print [np.linalg.norm(m) for m in mps1], [np.linalg.norm(m) for m in mps2]

    #     raise RuntimeError('error in mps dot')

    return E

    

def norm(mpx,direction=0,as_array=False): 
    """
    2nd norm of a MPX

    Parameters
    ----------
    mpx : MPS or MPO

    Returns
    -------
    norm : scalar
    """
    ovlp_val = mps_dot(np.conj(flatten(mpx)),flatten(mpx),direction=direction)

    # print ovlp_val.shape
    # print np.einsum('llrr->', ovlp_val)
    norm_val = np.sqrt(np.einsum('llrr->', ovlp_val))    # norm val is lLrR
    # ovlp_sq = tf.reshape(ovlp_val.transpose(0,2,1,3),'ii,ii')
    # norm_val = np.sqrt(np.einsum('xx->',ovlp_sq))

    # if len(mpx) < 3:
    #     print 'norm'
    #     print norm_val
    #     print np.linalg.norm(mpx.getSites())
    #     print np.sqrt( np.sum( np.abs( mpx.getSites() )**2 ) )

    #     m1x = mpx.getSites()
    #     normx = np.tensordot( np.conj(m1x), m1x, axes=(range(1,m1x.ndim-1),range(1,m1x.ndim-1)))
    #     print np.sqrt(np.einsum('lrlr->',normx))

    #     temp = mps_dot(np.conj(flatten(mpx)),flatten(mpx),direction=(direction+1)%2)
    #     print np.sqrt(np.einsum('llrr->', ovlp_val))


    # if not as_array:
    #     # catch cases when norm is ~0 but in reality is a small negative number
    #     assert(np.abs(np.imag(norm_val)/(np.real(norm_val)+1.0e-6)) < 1.0e-6 or np.imag(norm_val)<1.0e-10), norm_val
    #     assert(np.real(norm_val) > -1.0e-10), norm_val

    # if np.isnan(norm_val):  print 'norm ovlp', ovlp_val

    return norm_val


def mul(alpha, mpx):
    """
    scales mpx by constant factor alpha (can be complex)
    """

    L = mpx.shape                            # can work with high-dimensional arrays
    new_mpx = np.empty(L,dtype=np.object)

    const = np.abs(alpha)**(1./np.prod(L))
    dtype = np.result_type(alpha,mpx[0])
    for idx, m in np.ndenumerate(mpx):
        new_mpx[idx] = np.array(m,dtype=dtype)*const

    # change sign as specified by alpha
    if dtype == int or dtype == float:
        phase = np.sign(alpha)
    elif dtype == complex:
        phase = np.exp(1j*np.angle(alpha))
    else:
        raise(TypeError), 'not valid datatype %s' %dtype

    new_mpx[0] *= phase

    return MPX(new_mpx)


def add(mpx1, mpx2, obc=(True,True)):
    """
    Direct sum of MPX's of the same shape
    obc:  if True, reshape MPX s.t. end dims are 1
    """
    L = len(mpx1)

    assert mpx1.shape==mpx2.shape,\
           'add error: need to have same shapes: (%d,%d)'%(len(mpx1),len(mpx2))
    assert np.all(mpx1.phys_bonds==mpx2.phys_bonds),\
           'add error: need to have same physical bond dimensions'

    new_mpx = np.empty(L, dtype=np.object)
    dtype = np.result_type(mpx1[0], mpx2[0])

    for i in range(L):
        sh1 = mpx1[i].shape
        sh2 = mpx2[i].shape

        l1,n1,r1 = sh1[0],np.prod(sh1[1:-1]),sh1[-1]
        l2,n2,r2 = sh2[0],np.prod(sh2[1:-1]),sh2[-1]

        if L == 1 and obc[0] and obc[1] and l1==l2 and r1==r2:
            new_site = mpx1[i] + mpx2[i]
        elif i == 0 and obc[0] and l1==l2:
            new_site = np.zeros((l1,n1,r1+r2),dtype=dtype)
            new_site[:,:,:r1] = mpx1[i].reshape(l1,n1,r1)
            new_site[:,:,r1:] = mpx2[i].reshape(l2,n2,r2)
            new_site = new_site.reshape((l1,)+sh1[1:-1]+(r1+r2,))
        elif i == L-1 and obc[-1] and r1==r2:
            new_site = np.zeros((l1+l2,n1,r1),dtype=dtype)
            new_site[:l1,:,:] = mpx1[i].reshape(l1,n1,r1)
            new_site[l1:,:,:] = mpx2[i].reshape(l2,n2,r2)
            new_site = new_site.reshape((l1+l2,)+sh1[1:-1]+(r1,))
        else:
            new_site = np.zeros((l1+l2,n1,r1+r2),dtype=dtype)
            new_site[:l1,:,:r1] = mpx1[i].reshape(l1,n1,r1)
            new_site[l1:,:,r1:] = mpx2[i].reshape(l2,n2,r2)
            new_site = new_site.reshape((l1+l2,)+sh1[1:-1]+(r1+r2,))

        new_mpx[i] = new_site.copy()

    # if obc[0]:
    #     sh1 = mpx1[0].shape
    #     sh2 = mpx2[0].shape

    #     if sh1[0] == 1 and sh2[0] == 1:
    #         sh3 = new_mpx[0].shape
    #         new_site = np.einsum('l...r->...r',new_mpx[0]).reshape((1,)+sh3[1:])
    #         new_mpx[0] = new_site.copy()
    # if obc[1]:
    #     sh1 = mpx1[-1].shape
    #     sh2 = mpx2[-1].shape
    #   
    #     if sh1[-1] == 1 and sh2[-1] ==1:  
    #         sh3 = new_mpx[-1].shape
    #         new_site = np.einsum('l...r->l...',new_mpx[-1]).reshape(sh3[:-1]+(1,))
    #         new_mpx[-1] = new_site.copy()

    return MPX(new_mpx)



def add_el(mpx1, mpx2):
    """
    Elemental addition
    """
    L = len(mpx1)
    assert len(mpx2)==L, 'mpx1 and mpx2 need to have same length'

    new_mpx = empty(mpx1.phys_bonds)
    for i in range(L):
        assert(mpx1[i].shape == mpx2[i].shape), 'mpx1, mpx2 need to have same shape'
        new_mpx[i] = mpx1[i]+mpx2[i]
        
    return new_mpx


def axpby(alpha,mpx1,beta,mpx2):
    """
    return (alpha * mpx1) + (beta * mpx2)
    alpha, beta are scalar; mps1,mps2 are ndarrays of tensors (MPXs)
    """

    mpx_new = add(mul(alpha,mpx1),mul(beta,mpx))
    return mpx_new


def dot(mpx1, mpx2):
    """
    Computes MPX1 * MPX2

    Parameters
    ----------
    mpx1: MPO or MPS
    mpx2: MPO or MPS

    Returns
    -------
    new_mpx : float or MPS or MPO
    """

    L = len(mpx1)
    assert len(mpx2)==L, '[dot]: lengths of mpx1 and mpx2 are not equal'
    new_mpx = np.empty(L, dtype=np.object)

    if np.all([mpx1[i].ndim == 3 and mpx2[i].ndim == 3 for i in range(L)]):
        return mps_dot(mpx1,mpx2)
    else:
        for i in range(L):
            tot = mpx1[i].ndim + mpx2[i].ndim
            ax1 = [0] + range(2,     mpx1[i].ndim) + [tot-2]
            ax2 = [1] + range(mpx1[i].ndim, tot-2) + [tot-1]
            ax2[1] = ax1[-2]   # contract vertical bonds (mpx1 down with mpx2 up)
            new_site = np.einsum(mpx1[i],ax1,mpx2[i],ax2)
            new_mpx[i] = tf.reshape(new_site,'ij,...,kl',group_ellipsis=False)

    return MPX(new_mpx)


# ### single site compression ###
# def dot_compress(mpx1,mpx2,DMAX,direction=0):
#     # returns mpx1*mpx2 (ie mpsx1 applied to mpsx2) in mpx form, with compression of each bond
# 
#     # print 'dotcompress', direction
# 
#     L = len(mpx1)
#     assert(len(mpx2)==L), 'lens %d, %d'%(len(mpx2),L)
#     new_mpx = np.empty(L,dtype=np.object)
#     errs = np.array([0.]*(L-1))
# 
#     if not direction == 0:
#         mpx1 = [np.swapaxes(m,0,-1) for m in mpx1[::-1]]   # taking the left/right transpose
#         mpx2 = [np.swapaxes(m,0,-1) for m in mpx2[::-1]]
#    
# 
#     if np.all([mpx1[i].ndim == 3 and mpx2[i].ndim == 3 for i in range(L)]):
#         return mps_dot(mpx1,mpx2)
#     else:
#         # site 0
#         tot = mpx1[0].ndim + mpx2[0].ndim
#         ax1 = [0] + range(2,     mpx1[0].ndim) + [tot-2]
#         ax2 = [1] + range(mpx1[0].ndim, tot-2) + [tot-1]
#         ax2[1] = ax1[-2]   # contract vertical bonds (mpx1 down with mpx2 up)
#         new_site = np.einsum(mpx1[0],ax1,mpx2[0],ax2)
#         new_site = tf.reshape(new_site,'ij,...',group_ellipsis=False)
#         # u,s,vt,dwt = tf.svd(new_site,'...,kl',DMAX)  # mpx2/mpx1 virtual bonds
#         u,s,vt,dwt = tf.svd(new_site,-2,DMAX)
#         new_mpx[0] = u.copy()
#         env        = tf.dMult('DM',s,vt)
#         errs[0]   += dwt
# 
#         # remaining sites
#         for i in range(1,L):
#             time1 = time.time()
#             tot = mpx1[i].ndim + mpx2[i].ndim
#             ax1 = [0] + range(2,     mpx1[i].ndim) + [tot-2]
#             ax2 = [1] + range(mpx1[i].ndim, tot-2) + [tot-1]
#             ax2[1] = ax1[-2]   # contract vertical bonds (mpx1 down with mpx2 up)
#             new_site = np.einsum(mpx1[i],ax1,mpx2[i],ax2)
#             new_site = np.einsum('abc,bc...->a...',env,new_site)
# 
#             if i == L-1:
#                 new_mpx[i] = tf.reshape(new_site,'...,kk',group_ellipsis=False)
#             else:
#                 u,s,vt,dwt = tf.svd(new_site,'...,kl',DMAX)
#                 new_mpx[i] = u.copy()
#                 errs[i]   += dwt
#                 env        = tf.dMult('DM',s,vt)
#                
# 
#     if not direction == 0:
#         new_mpx = [np.swapaxes(m,0,-1) for m in new_mpx[::-1]]   # taking the left/right transpose
#         errs    = errs[::-1]
# 
#     # print [s.shape[0] for s in new_mpx]
# 
#     return MPX(new_mpx), errs

### two-site compression
# @profile
def dot_compress(mpx1,mpx2,DMAX,direction=0):
    # returns mpx1*mpx2 (ie mpsx1 applied to mpsx2) in mpx form, with compression of each bond

    # print 'dotcompress', direction

    L = len(mpx1)
    assert(len(mpx2)==L), 'lens %d, %d'%(len(mpx2),L)
    new_mpx = np.empty(L,dtype=np.object)
    errs = np.array([0.]*(L-1))

    if not direction == 0:
        mpx1 = [np.swapaxes(m,0,-1) for m in mpx1[::-1]]   # taking the left/right transpose
        mpx2 = [np.swapaxes(m,0,-1) for m in mpx2[::-1]]
   

    if np.all([mpx1[i].ndim == 3 and mpx2[i].ndim == 3 for i in range(L)]):
        return vdot(np.conj(mpx1),mpx2)    # bc vdot takes cc
    else:
        # site 0
        tot = mpx1[0].ndim + mpx2[0].ndim
        ax1 = [0] + range(2,     mpx1[0].ndim) + [tot-2]
        ax2 = [1] + range(mpx1[0].ndim, tot-2) + [tot-1]
        ax2[1] = ax1[-2]   # contract vertical bonds (mpx1 down with mpx2 up)
        site0 = np.einsum(mpx1[0],ax1,mpx2[0],ax2)
        site0 = tf.reshape(site0,'ij,...,jk',group_ellipsis=False)


        # for remaining sites
        for i in range(1,L):
            tot = mpx1[i].ndim + mpx2[i].ndim
            ax1 = [0] + range(2,     mpx1[i].ndim) + [tot-2]
            ax2 = [1] + range(mpx1[i].ndim, tot-2) + [tot-1]
            ax2[1] = ax1[-2]   # contract vertical bonds (mpx1 down with mpx2 up)

            # print mpx1[i].shape, ax1, mpx2[i].shape, ax2

            site1 = np.einsum(mpx1[i],ax1,mpx2[i],ax2)
            site1 = tf.reshape(site1, 'ij,...,jk')

            two_site = np.tensordot(site0,site1,axes=[-1,0])
            cut_pos = site0.ndim-1
            u,s,vt,dwt = tf.svd(two_site,cut_pos,DMAX)
            new_mpx[i-1] = u.copy()
            errs[i-1]   += dwt
            site0        = tf.dMult('DM',s,vt)

        new_mpx[-1] = site0.copy()
        errs[-1]   += dwt

    if not direction == 0:
        new_mpx = [np.swapaxes(m,0,-1) for m in new_mpx[::-1]]   # taking the left/right transpose
        errs    = errs[::-1]

    # print [s.shape[0] for s in new_mpx]

    return MPX(new_mpx), errs


def dot_block(mpx,ind0,numSites,MPOblock,block_order='io',direction=0,DMAX=100,compress=True,
              normalize=False):
    """
    apply operator MPOblock (ndarray) onto network, compressing at the end
    assumes all physical bonds are correctly grouped into up/down bonds

    Parameters:
    -----------
    ind0:        first site the block is applied onto
    numSites:    number of sites the block spans
    MPOblock:    block to be applied
    block_order: ordering of block legs
                 'io':  1 x (d_out x d_out ...) x (d_in x d_in x ...) x 1
                 'site: 1 x (d_out x d_in) x (d_out x d_in) ... x 1

    """  
    assert(isinstance(MPOblock,np.ndarray)), 'MPOblock needs to be ndarray'

    ## first contract all sites into block (bc physical bonds usually < virtual bond)
    ## then compress block into MPX

    ##### contraction step #####
    block = MPOblock.copy()
    if block_order == 'site':
        tot_ndp = 0
        for i in range(numSites):
            ii = ind0 + i
            # n_dp  = mpx[ii].ndim-2  # 1 if mps, 2 if mpo
            site_label = [0] + range(2,mpx[ii].ndim) + [4] 
                                                 # l u d r -> 0 2 3 4  (MPO)
                                                 # l u   r -> 0 2   4  (MPS)
            if i == 0:
                block_label = range(-1,0) + [1, 2] + range(5,5+(numSites-i-1)*2+1)
                                                 # l u d ... for site 0
                                                 # ... d-1 u d ... for site > 0
            else:
                # block_label = range(-i*n_dp-3,0) + [0, 1, 2] + range(5,5+(numSites-i-1)*2+1)
                block_label = range(-tot_ndp-2,0) + [0, 1, 2] + range(5,5+(numSites-i-1)*2+1)
                # block_label = range(-2*i-2,0) + [0, 1, 2] + range(5,5+(numSites-i-1)*2+1)
        
            tot_ndp += mpx[ii].ndim-2  # 1 if mps, 2 if mpo

            # print site_label, block_label
            # print mpx[ii].shape, block.shape 
               
            offset = np.min(site_label+block_label)
            site_label  = [sl - offset for sl in site_label]
            block_label = [bl - offset for bl in block_label]
            block = np.einsum(block,block_label,mpx[ii],site_label)

        block = tf.reshape(block,'ij,...,kl', group_ellipsis=False)

    elif block_order == 'io':
        for i in range(numSites):
            ii = ind0 + i
            site_label = [0] + range(1,mpx[ii].ndim) 
                                                 # l u d r -> 0 1 2 3  (MPO)
                                                 # l u   r -> 0 1   2  (MPS)
            if i == 0:   block_label = [0] + range(-numSites,0) + [1] + range(4,4+(numSites-i-1)+1)
                                                 # l (u...) d .... 
            else:        block_label = range(-numSites-1-i,0) + [0,1] + range(4,4+(numSites-i-1)+1)
                                                 # . (u...) ... l d ...
 
            offset = np.min(site_label+block_label)
            site_label  = [sl - offset for sl in site_label]
            block_label = [bl - offset for bl in block_label]
            block = np.einsum(block,block_label,mpx[ii],site_label)
    else:
        raise(ValueError), 'block_order should be io or site'

    if normalize:
        bl_norm = np.linalg.norm(block)
        block = block*(1./bl_norm)

    if compress:
        ##### compression step #####
        # print ind0a
        n_dp = mpx[i].ndim-2  # 1 if mps, 2 if mpo
        if direction== 0:   svd_str = n_dp+1
        else:               svd_str = -(n_dp+1)
        new_sites = tf.decompose_block(block, numSites, direction, DMAX, svd_str=svd_str)
        new_sites = MPX(new_sites)

        # new_sites = empty(mpx.phys_bonds[ind0:ind0+numSites])
        # if direction == 0:   # left to right
        #     for i in range(numSites-1):
        #         n_dp = mpx[i].ndim-2  # 1 if mps, 2 if mpo
        #         u, s, vt, dwt = tf.svd(block,n_dp+1,DMAX)
        #         new_sites[i] = u
        #         block = tf.dMult('DM',s,vt)
        #     new_sites[-1] = block
        # elif direction == 1:
        #     for i in range(1,numSites)[::-1]:
        #         n_dp = mpx[i].ndim-2  # 1 if mps, 2 if mpo
        #         n_bl = block.ndim
        #         u, s, vt, dwt = tf.svd(block,n_bl-n_dp-1,DMAX)
        #         new_sites[i] = vt
        #         block = tf.dMult('MD',u,s)
        #     new_sites[0] = block
        # else:
        #     raise ValueError, 'direction is 0 (L->R) or 1 (R->L)'

        new_array = np.append(mpx[:ind0],np.append(new_sites,mpx[ind0+numSites:]))

        return MPX(new_array, phys_bonds=mpx.phys_bonds)
    else:  
        return block.copy()


def dot_block_block(block1,block2,in_order=('site','site'),out_order='site'):
    
     L = (len(block1.shape)-2)/2
     assert(block1.shape == block2.shape), 'dot_block_block:  block shapes are different'
     sqdim = int(np.sqrt(np.prod(block1.shape)))

     axTio = tf.site2io(L)
     axTst = tf.io2site(L)

     if in_order[0] == 'site':
         b1 = block1.transpose(axTio)
         orig_shape = b1.shape
         b1 = b1.reshape(sqdim,sqdim)
     else:
         orig_shape = b1.shape
         b1 = block1.reshape(sqdim,sqdim)
         
     if in_order[1] == 'site':         b2 = block2.transpose(axTio).reshape(sqdim,sqdim)
     else:                             b2 = block2.reshape(sqdim,sqdim)

     b_out = np.matmul(b1,b2)
     b_out = b_out.reshape(orig_shape)
 
     if out_order == 'site':    b_out = b_out.transpose(axTst)

     return b_out
 
    
def outer(mpx1,mpx2):
    """
    takes the outer product of two MPX (ndim = 3) -? |MPX1><MPX2|
    take complex conj and transposeUD of mpx if needed
    """
    assert(len(mpx1)==len(mpx2)), 'need MPS of the same lenght, now %d, %d'%(len(mpx1),len(mpx2))

    L = len(mpx1)
    new_sites = []

    
    for i in range(L):
        sh1 = mpx1[i].shape
        sh2 = mpx2[i].shape
        nd1 = len(sh1)
        nd2 = len(sh2)

        site = np.outer(mpx1[i],mpx2[i]).reshape(sh1+sh2)
        axT  = [0,nd1] + range(1,nd1-1) + range(nd1+1,nd1+nd2-1)[::-1] + [nd1-1,nd1+nd2-1]
        site = tf.reshape(site.transpose(axT),'ij,...,jk',group_ellipsis=False)
             ## eg. two MPS:  lar/LAR   -> lL x a x A x rR
             ## eg. two MPO:  labr/LABR -> lL x a x b x B x A x rR
        new_sites.append(site)

    try:
        new_dp = np.empty(L,dtype=object)
        new_dp = [mpx1.phys_bonds[i] + mpx2.phys_bonds[i] for i in range(L)]
    except(AttributeError):  #mpx1, mpx2 aren't MPX objects
        new_dp = None
    
    return MPX(new_sites,new_dp)




#################################
####   other MPX functions   ####
#################################

    
def compress_1(mpx0, D, direction=0):
    """
    compress MPX to max bond dimension D, one site at a time

    Return
    -----
    compresed MPX
    errors resulting from compression at each bond (bond follows site)
    """

    L = len(mpx0)
    errs = np.array([0.]*(L-1))

    mpx = mpx0.copy()
    
    if direction == 0:
        for i in range(L-1):
            u, s, vt, dwt = tf.svd(mpx[i], "...,k", D)
            # print 'compress f', i, mpx[i].shape, u.shape, vt.shape
            mpx[i] = u
            # svt = np.dot(np.diag(s), vt)
            svt = tf.dMult('DM',s,vt)
            # try:
            mpx[i+1] = np.einsum("lj,j...r->l...r", svt, mpx[i+1])
            errs[i] += dwt    # assume that it's 0 for obc
            # except(IndexError):   # i = L-1
            #     mpx[i]   = np.einsum('ij,j...r->l...r',mpx[i],svt)
    else:
        for i in range(L-1,0,-1):
            u, s, vt, dwt = tf.svd(mpx[i], "i,...", D)
            # print 'compress b', i, mpx[i].shape, u.shape, vt.shape
            mpx[i] = vt
            # us = np.dot(u,np.diag(s))
            us = tf.dMult('MD',u,s)
            # try:
            mpx[i-1] = np.einsum("l...j,jr->l...r", mpx[i-1], us)
            errs[i-1] += dwt   # assume that it's 0 for obc
            # except(IndexError):  pass   # i = 0

    return mpx, errs


# @profile
def compress(mpx0, D, direction=0, ref_bl=None, tol=1.0e-8):
    """
    compress MPX to max bond dimension D, two sites at a time

    Return
    ------
    compresed MPX
    errors resulting from compression at each bond (bond follows site)
    """

    L = len(mpx0)
    errs = np.array([0.]*(L-1))

    mpx = MPX.copy(mpx0)
    # mpx = mpx0.copy()

    if direction == 0:        check_canon(mpx,canon=1,ind0=1)
    else:                     check_canon(mpx,canon=0,ind0=L-1)
    
    # print 'compress'

    if direction == 0:
        for i in range(L-1):
            tens_block = np.tensordot(mpx[i],mpx[i+1],axes=(-1,0))

            # if ref_bl is not None:
            #     pass
            #     # print 'compress compare', np.linalg.norm(ref_bl-tens_block)
            #     # tens_block = ref_bl

            b_ind = mpx[i].ndim-1
            u, s, vt, dwt = tf.svd(tens_block, b_ind, D, tol=tol)
            errs[i] += dwt
            mpx[i]   = u
            mpx[i+1] = tf.dMult('DM',s,vt)
            # print s, u.shape, vt.shape
    else:
        for i in range(L-1,0,-1):
            tens_block = np.tensordot(mpx[i-1],mpx[i],axes=(-1,0))
            b_ind = mpx[i-1].ndim-1
            u, s, vt, dwt = tf.svd(tens_block, b_ind, D, tol=tol)
            errs[i-1] += dwt
            mpx[i]     = vt
            mpx[i-1]   = tf.dMult('MD',u,s)
 
    # print errs
    # print MPX.norm( mpx0-mpx )
    # print np.linalg.norm( mpx0.getSites() - mpx.getSites() )
    # print [m.shape for m in mpx]

    return mpx, errs


def canonicalize(mpx0, direction=0, use_qr=False):
    ''' put mps in right/left canonicalization using QR/LQ
    '''

    L = len(mpx0)
    mpx = copy(mpx0)

    # print 'canonicalize', [m.shape for m in mpx]

    if direction==0:
        for i in range(L-1):
            b_ind = mpx[i].ndim-1
            if use_qr:
                q,r = tf.qr(mpx[i], b_ind)
                mpx[i] = q
                mpx[i+1] = np.tensordot(r,mpx[i+1],axes=(-1,0))
            else:
                u, s, vt, dwt = tf.svd(mpx[i],b_ind,DMAX=-1,tol=None)
                mpx[i] = u
                mpx[i+1] = np.tensordot( tf.dMult('DM',s,vt), mpx[i+1], axes=(-1,0) )
            

            # tens_block = np.tensordot(mpx[i],mpx[i+1],axes=(-1,0))
            # q,r = tf.qr(tens_block, b_ind)
            # mpx[i]   = q
            # mpx[i+1] = r
    else:
        for i in range(L-1,0,-1):
            if use_qr:
                l,q = tf.lq(mpx[i], 1)
                mpx[i] = q
                mpx[i-1] = np.tensordot(mpx[i-1],l,axes=(-1,0))
            else:
                u, s, vt, dwt = tf.svd(mpx[i],1,DMAX=-1,tol=None)
                mpx[i] = vt
                mpx[i-1] = np.tensordot( mpx[i-1], tf.dMult('MD',u,s), axes=(-1,0) )

            # tens_block = np.tensordot(mpx[i-1],mpx[i],axes=(-1,0))
            # b_ind = mpx[i-1].ndim-1
            # l,q = tf.lq(tens_block, b_ind)
            # mpx[i-1]   = l
            # mpx[i]     = q

    return mpx


def compress_reg(mpx0, DMAX=-1, direction=0, tol=1.0e-8):
    ''' do svd but split sing vals across adjacent matrices 
        default is to not do any compression (so that there's no bad truncation err)
    '''
    L = len(mpx0)
    errs = np.array([0.]*(L-1))
    s_list = np.empty(L-1,dtype=list)

    mpx = MPX.copy(mpx0)
    # mpx = mpx0.copy()

    if direction == 0:        check_canon(mpx,canon=1,ind0=1)
    else:                     check_canon(mpx,canon=0,ind0=L-1)
    
    if direction == 0:
        for i in range(L-1):
            tens_block = np.tensordot(mpx[i],mpx[i+1],axes=(-1,0))
            b_ind = mpx[i].ndim-1
            u, s, vt, dwt = tf.svd(tens_block, b_ind, DMAX, tol=tol)
            errs[i] += dwt
            s_list[i] = s
            mpx[i]   = u
            mpx[i+1] = tf.dMult('DM',s,vt)

            # s_list[i]= s[s>tol]            
            # M = len(s_list[i])
            # if M == 0:  print 'M = 0'
            # utemp = np.moveaxis(u,-1,0)
            # utemp = utemp[:M]
            # mpx[i]   = np.moveaxis(utemp,0,-1)
            # mpx[i+1] = tf.dMult('DM',s[:M],vt[:M])

        ## regularization
        mpx_norm = np.linalg.norm(mpx[-1]) # norm(mpx)
        # print 'altD norm', mpx_norm, np.linalg.norm(mpx[-1])
        mpx[-1] = mpx[-1]*1./mpx_norm
        mpx = mul(mpx_norm,mpx)

        # for i in range(L-1):

        #     mpx[i] = tf.dMult('MD',mpx[i],np.sqrt(s_list[i]))
        #     if not s_list[i][0] == 0.:
        #         s_inv = 1./s_list[i]
        #         mpx[i+1] = tf.dMult('DM',np.sqrt(s_inv),mpx[i+1])
        #     else:
        #         mpx[i+1] = mpx[i+1]*0.  # 0

    else:
        for i in range(L-1,0,-1):
            tens_block = np.tensordot(mpx[i-1],mpx[i],axes=(-1,0))
            b_ind = mpx[i-1].ndim-1
            u, s, vt, dwt = tf.svd(tens_block, b_ind, DMAX, tol=tol)
            errs[i-1] += dwt
            s_list[i-1]= s
            mpx[i]     = vt
            mpx[i-1]   = tf.dMult('MD',u,s)

        ## regularize
        mpx_norm = np.linalg.norm(mpx[0])  #norm(mpx)
        # print 'altD norm', mpx_norm, np.linalg.norm(mpx[0])
        mpx[0] = mpx[0]*1./mpx_norm
        mpx = mul(mpx_norm,mpx)

        # for i in range(L-1,0,-1):
        #     mpx[i] = tf.dMult('DM',np.sqrt(s_list[i-1]),mpx[i])
        #     if not s_list[i-1][0] == 0.:
        #         s_inv = 1./s_list[i-1]
        #         mpx[i-1] = tf.dMult('MD',mpx[i-1],np.sqrt(s_inv))
        #     else:
        #         mpx[i-1] = mpx[i-1]*0.   # 0

    # print 'mpx regularize'
    # print [np.linalg.norm(m) for m in mpx0]
    # print [np.linalg.norm(m) for m in mpx]
    # print MPX.norm( mpx0 - mpx )

    return mpx, errs


def regularize(mpx):

    mpx_ = copy(mpx)

    for i in range(len(mpx)):
        scale = 1. / np.linalg.norm(mpx[i])
        mpx_[i] = scale*mpx[i]

    norm1 = norm(mpx)
    norm2 = norm(mpx_)
    mpx_ = mul(norm1/norm2,mpx_)

    # print 'regularize', norm(mpx), norm(mpx_), norm1, norm2, [np.linalg.norm(m) for m in mpx_]

    return mpx_
        


# compress bond to bond dims D1, D2 in alternating fashion
def altD_compress(mpx0,Ds,direction,regularize=True):

    L = len(mpx0)
    num_Ds = len(Ds)
    errs = np.array([0.]*(L-1))
    s_list = np.empty(L-1,dtype=list)

    mpx = copy(mpx0)
    # mpx = mpx0.copy()

    if direction == 0:        check_canon(mpx,canon=1,ind0=1)
    else:                     check_canon(mpx,canon=0,ind0=L-1)
    
    if direction == 0:
        for i in range(L-1):
            tens_block = np.tensordot(mpx[i],mpx[i+1],axes=(-1,0))
            b_ind = mpx[i].ndim-1
            u, s, vt, dwt = tf.svd(tens_block, b_ind, Ds[i%num_Ds])
            errs[i] += dwt
            s_list[i] = s
            mpx[i]   = u
            mpx[i+1] = tf.dMult('DM',s,vt)

        if regularize:
            mpx_norm = np.linalg.norm(mpx[-1]) # norm(mpx)
            # print 'altD norm', mpx_norm, np.linalg.norm(mpx[-1])
            mpx[-1] = mpx[-1]*1./mpx_norm
            mpx = mul(mpx_norm,mpx)

            # for i in range(L-1):
            #     mpx[i] = tf.dMult('MD',mpx[i],np.sqrt(s_list[i]))
            #     if not s_list[i][0] == 0.:
            #         s_inv = 1./s_list[i]
            #         mpx[i+1] = tf.dMult('DM',np.sqrt(s_inv),mpx[i+1])
            #     else:
            #         mpx[i+1] = mpx[i+1]*0.  # 0

    else:
        for i in range(L-1,0,-1):
            tens_block = np.tensordot(mpx[i-1],mpx[i],axes=(-1,0))
            b_ind = mpx[i-1].ndim-1
            u, s, vt, dwt = tf.svd(tens_block, b_ind, Ds[i%num_Ds])
            errs[i-1] += dwt
            s_list[i-1]= s
            mpx[i]     = vt
            mpx[i-1]   = tf.dMult('MD',u,s)
       
        if regularize:
            mpx_norm = np.linalg.norm(mpx[0])  #norm(mpx)
            # print 'altD norm', mpx_norm, np.linalg.norm(mpx[0])
            mpx[0] = mpx[0]*1./mpx_norm
            mpx = mul(mpx_norm,mpx)

            # for i in range(L-1,0,-1):
            #     mpx[i] = tf.dMult('DM',np.sqrt(s_list[i-1]),mpx[i])
            #     if not s_list[i-1][0] == 0.:
            #         s_inv = 1./s_list[i-1]
            #         mpx[i-1] = tf.dMult('MD',mpx[i-1],np.sqrt(s_inv))
            #     else:
            #         mpx[i-1] = mpx[i-1]*0.   # 0

    # print 'mpx regularize'
    # print [np.linalg.norm(m) for m in mpx0]
    # print [np.linalg.norm(m) for m in mpx]
    # print MPX.norm( mpx0 - mpx )

    return mpx, errs


# def split_singular_vals(mpx_list, s_list, direction=0):
# 
#     ''' direction is canonicalization direction of pepx_list
#         len s_vals = len(mpx)-1
#     '''
# 
#     # split singular values
#     reg_list = []
# 
#     mtens = mpx_list[0]
#     if direction == 0:      mtens = tf.dMult('MD',mtens,np.sqrt(s_list[0]))
#     elif direction == 1:    mtens = tf.dMult('MD',mtens,1./np.sqrt(s_list[0]))
#     reg_list.append(mtens)
# 
#     for ind in range(1,len(mpx_list)-1):
#         mtens = mpx_list[ind]
#         if direction == 0:
#             mtens = tf.dMult('DM',1./np.sqrt(s_list[ind-1]),mtens)
#             mtens = tf.dMult('MD',mtens,np.sqrt(s_list[ind]))
#         elif direction == 1:
#             mtens = tf.dMult('DM',np.sqrt(s_list[ind-1]),mtens)
#             mtens = tf.dMult('MD',mtens,1./np.sqrt(s_list[ind]))
#         else:  
#             print 'provide valid direction, not ', direction
#             exit()
#         reg_list.append(mtens)
# 
#     mtens = mpx_list[-1]
#     if direction == 0:      ptens = tf.dMult('DM',1./np.sqrt(s_list[-1]),mtens)
#     elif direction == 1:    ptens = tf.dMult('DM',np.sqrt(s_list[-1]),mtens)
#     reg_list.append(mtens)
# 
#     return MPX(reg_list)
         

def inprod(mps1, mpo, mps2, direction=0):
    """
    Computes <MPS1 | MPO | MPS2>
    
    Note: bra is not conjugated, and
          MPS1, MPS2 assumed to have OBC

    Parameters
    ----------
    mps1 : MPS
    mpo : MPO
    mps2 : MPS

    Returns
    -------
    inprod : float

    """
    assert direction in (0, 1)
    L = len(mps1)
    
    if direction==0:
        left = np.einsum('lnr,anNb,LNR->laLrbR', np.conj(mps1[0]), mpo[0], mps2[0])
        # print left.shape
        for i in range(1,L):
            temp = np.einsum('...Lal,audb->...Ludbl', left, mpo[i])
            temp = np.einsum('...Ludbl,LuR->...Rdbl', temp, np.conj(mps1[i]))
            temp = np.einsum('...Rdbl,ldr->...Rbr', temp, mps2[i])
            left = temp
        return np.squeeze(left)
        
    elif direction==1:
        right = np.einsum('lnr,anNb,LNR->laLrbR', np.conj(mps1[-1]), mpo[-1], mps2[-1])
        for i in range(0,L-1)[::-1]:
            temp = np.einsum('audb,Rbr...->Raudr...', mpo[i], right)
            temp = np.einsum('LuR,Raudr...->Ladr...', np.conj(mps1[i]), temp)
            temp = np.einsum('ldr,Ladr...->Lal...', mps2[i], temp)
            right = temp
        return np.squeeze(right)
 
    else:
        print 'choose direction 0 or 1'


def inprod_anc(mps1,mpo,mps2,direction=0):

    L = len(mps1)

    if direction==0:
        # left = np.einsum('lnxr,anNb,LNxR->laLrbR', np.conj(mps1[0]), mpo[0], mps2[0])
        left = np.einsum('lnxr,anNb->laxNrb', np.conj(mps1[0]), mpo[0])
        left = np.einsum('laxNrb,LNxR->laLrbR', left, mps2[0])
        # print left.shape
        for i in range(1,L):
            temp = np.einsum('...Lal,audb->...Ludbl', left, mpo[i])
            temp = np.einsum('...Ludbl,LuxR->...Rdxbl', temp, np.conj(mps1[i]))
            temp = np.einsum('...Rdxbl,ldxr->...Rbr', temp, mps2[i])
            left = temp
        return np.squeeze(left)
        
    elif direction==1:
        # right = np.einsum('lnxr,anNb,LNxR->laLrbR', np.conj(mps1[-1]), mpo[-1], mps2[-1])
        right = np.einsum('lnxr,anNb->laxNrb', np.conj(mps1[-1]), mpo[-1])
        right = np.einsum('laxNrb,LNxR->laLrbR', right, mps2[-1])
        for i in range(0,L-1)[::-1]:
            temp = np.einsum('audb,Rbr...->Raudr...', mpo[i], right)
            temp = np.einsum('LuxR,Raudr...->Ladxr...', np.conj(mps1[i]), temp)
            temp = np.einsum('ldxr,Ladxr...->Lal...', mps2[i], temp)
            right = temp
        return np.squeeze(right)
 
    else:
        print 'choose direction 0 or 1'


def measMPO(mps,ind0,L,mpo):
    mps_obs = mps.copy()
    mps_obs[ind0:ind0+L] = MPX.dot(mpo, mps_obs[ind0:ind0+L])
    return MPX.dot(np.conj(mps),mps_obs)

    
def flatten(mpx):
    """
    Converts MPX object into MPS

    Parameters
    ----------
    mpx : MPS or MPO

    Returns
    -------
    mps : 
    """
    if np.all( [mpx[x].ndim == 3 for x in range(len(mpx))] ): # already MPS
        return mpx
    else: # MPO
        # assert mpx[0].ndim == 4
        L = len(mpx)
        mps = []
        for i in range(L):
            sh = mpx[i].shape
            mps.append(tf.reshape(mpx[i],[1,-1]))
            # mps.append(np.reshape(mpx[i], (sh[0], sh[1]*sh[2], -1)))

        return MPX(mps)


def transposeUD(mpx):
    ## swap "up" and "down", "in" and "out" bonds
    ## uo x do x di x ui --> ui x di x do x uo
    ## up x down         --> down x up

    L = len(mpx)
    new_sites = []

    for ii in range(L):

        axT = [0] + range(1,mpx[ii].ndim-1)[::-1] + [-1]
        new_sites.append(mpx[ii].transpose(axT))

    return MPX(new_sites)


# def orthog_MPO(mpo):
#     """
#     finds orthogonal state for entire MPO
#     if end dimensions of MPO are not 1, find orthog state for each virtual bond of MPO
#     put all back into 
# 
#     """
# 
#     dp_ = [d[0] for d in mpo.phys_bonds]
#     iden_mpo = eye(dp_)
# 
#     bdim = mpo[0].shape[0]
#     if bdim == 1:
#         return add(iden_mpo,mul(-1,mpo))
#     else:
#         neg_mpo  = mul(-1,mpo)
#         o_mpo    = neg_mpo.copy()
#         o_mpo[0] = neg_mpo[0][0:1]
# 
#         for i in range(1,bdim):
#             mpo_1    = neg_mpo.copy()
#             mpo_1[0] = neg_mpo[0][i:i+1]
# 
#             temp = add(iden_mpo,mpo_1)
#             o_mpo = compress(add(o_mpo,temp,obc=(False,True))
# 
#         return o_mpo   
 
       
def orthog_MPO(mpo):
    """
    finds orthogonal state for entire MPO
    if end dimensions of MPO are not 1, find orthog state for each virtual bond of MPO
    put all back into 

    """

    dp_ = [d[0] for d in mpo.phys_bonds]
    iden_mpo = eye(dp_)

    # bdim = mpo[0].shape[0]
    # if bdim == 1:
    return add(iden_mpo,mul(-1,mpo))
    # else:
    #     neg_mpo  = mul(-1,mpo)
    #     o_mpo    = neg_mpo.copy()
    #     o_mpo[0] = neg_mpo[0][0:1]

    #     for i in range(1,bdim):
    #         mpo_1    = neg_mpo.copy()
    #         mpo_1[0] = neg_mpo[0][i:i+1]

    #         temp = (iden_mpo,mpo_1)
    #         o_mpo = compress(add(o_mpo,temp,obc=(False,True)),-1)[0]
    #     
    #     return o_mpo   


def getSites(mpx,ind0,numSites):
    # returns ndarray
    block = mpx[ind0]
    for i in range(1,numSites):
        block = np.tensordot(block, mpx[ind0+i], axes=(-1,0))
    return block


def normalize(mpx,target_norm=1.):
    norm = mpx.norm()
    return mul(target_norm/norm, mpx)


def inverse(mpo, direction=0, DMAX=100, order=0):
    assert(np.all([len(ds) == 2 for ds in mpo.phys_bonds])), 'can only invert mpo'
    numSites = len(mpo)

    if order == 0:  # exact inverse
        block = getSites(mpo,0,numSites)
        axT = tf.site2io(numSites)
        sqdim = int( np.sqrt(np.prod(mpo.phys_bonds.tolist())) )
        io_block = block.transpose(axT)
        io_shape = io_block.shape
        io_block = io_block.reshape(sqdim,sqdim)

        inv_io   = np.linalg.inv(io_block)
        axT = tf.io2site(numSites)
        inv_block = inv_io.reshape(io_shape).transpose(axT)

        ##### compression step #####
        inv_sites = empty(mpo.phys_bonds)
        block = inv_block
        if direction == 0:   # left to right
            for i in range(1,numSites):
                n_dp = mpo[i].ndim-2  # 1 if mps, 2 if mpo
                u, s, vt, dwt = tf.svd(block,3,DMAX)
                inv_sites[i-1] = u
                block = tf.dMult('DM',s,vt)
            inv_sites[-1] = block
        elif direction == 1:
            for i in range(1,numSites)[::-1]:
                n_dp = mpo[i].ndim-2  # 1 if mps, 2 if mpo
                u, s, vt, dwt = tf.svd(block,3,DMAX)
                inv_sites[i] = vt
                block = tf.dMult('MD',u,s)
            inv_sites[0] = block
        else:
            raise ValueError, 'direction is 0 (L->R) or 1 (R->L)'
        return MPX(inv_sites)
    else:
        inv = MPX.eye([d[0] for d in mpo.phys_bonds])
        mpo_pow = mpo.copy()
        inv = inv + mpo_pow

        step = 1
        while step < order:
            mpo_pow = MPX.dot_compress( mpo, mpo_pow )[0]
    	    inv = inv + mpo_pow

        return inv


def getSingVals_1(mpx,direction=0):

    mpx_ = mpx.copy()
    sNorm = []
    
    if direction == 0:
        mpx_ = compress(mpx_,-1,1)[0]
        for i in range(len(mpx)):
            u,s,vt,dwt = tf.svd(mpx_[i],'...,k')
            svt = tf.dMult('DM',s,vt)
            sNorm.append(np.sum(s**2))

            if i==0:
                tr_ = np.einsum('ljr,ljR->rR',u,np.conj(u))
                print tr_
                print s, np.sum(s**2)
            else:
                tr_ = np.einsum('lL,ljr,ljR->rR',tr_,u,np.conj(u))
                print 'i', i
                print 'tr', tr_
                print 's', s**2

            try:                 mpx_[i+1] = np.einsum('ab,b...->a...',svt,mpx_[i+1])
            except(IndexError):  pass
        return sNorm
    else:
        mpx_ = compress(mpx_,-1,0)[0]
        for i in range(len(mpx))[::-1]:
            u,s,vt,dwt = tf.svd(mpx_[i],'j,...')
            mpx_[i] = vt.copy()

            if i==len(mpx)-1:
                tr_ = np.einsum('ljr,Ljr->lL',vt,np.conj(vt))
                print 'tr', tr_
                print 's', s**2, (s**2).shape
                print s, np.sum(s**2)
            else:
                tr_ = np.einsum('ljr,LjR,rR',vt,np.conj(vt),tr_)
                print 'i', i
                print 'tr', tr_
                print 's', s**2

                


            us = tf.dMult('MD',u,s)
            # assert(np.allclose(us,np.einsum('ijk,kl->ijl',u,np.diag(s)))), 'bad dMult'
            sNorm.append(np.sum(s**2))
            try:                 mpx_[i-1] = np.einsum('...a,ab->...b',mpx_[i-1],us)
            except(IndexError):  pass
        return sNorm[::-1]
 
      
def getSingVals_2(mpx,direction=0):

    mpx_ = mpx.copy()
    sNorm = []
    
    if direction == 0:
        for i in range(len(mpx)-1):
            mpx_block  = getSites(mpx_,i,2)
            u,s,vt,dwt = tf.svd(mpx_block,'...,jk')
            sNorm.append(np.sum(s**2))
            mpx_[i+1]  = tf.dMult('DM',s,vt)
        return sNorm
    else:
        for i in range(1,len(mpx))[::-1]:
            mpx_block  = getSites(mpx_,i-1,2)
            u,s,vt,dwt = tf.svd(mpx_block,'jk,...')
            sNorm.append(np.sum(s**2))
            mpx_[i-1]  = tf.dMult('MD',u,s)
        return sNorm[::-1]

