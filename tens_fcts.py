import re
import numpy as np
import scipy, scipy.linalg


tol = 1.0e-8

def idx_interp(idx,checkDim):
    """
    index interpretation (for splitting, reshaping tensors)
    
    idx:  string of desired axes
          or list of integers specifying axes to cut
    
    Returns
    -------
    int_idx:  list of integers specifying at which axes the splits occur
    """

####    >>>>>>>>>.... need to catch case where idx > ndim ........<<<<<<<<<<<<<<

    if isinstance(idx,str):
        idx0 = re.split(",", idx)
        ellipsis = [x == '...' for x in idx0]
        splits   = [len(x) for x in idx0]   # num characters between commas
        L = len(splits)

        try:
            mid = np.where(ellipsis)[0][0]
        except(IndexError):
            mid = L-1
            assert(np.sum(splits)==checkDim), 'str doesnt specify all axes of input'

        ind = 0
        cutsLR = np.cumsum(splits[:mid],dtype=np.int)
        cutsRL = np.cumsum(splits[:mid:-1],dtype=np.int)[::-1]
        cutInds = np.append(cutsLR,-1*cutsRL)
    elif isinstance(idx,int):
        cutInds = [idx]
        ellipsis = [False,False]
    elif np.all([isinstance(ix,int) for ix in idx]):
        cutInds = idx
        ellipsis = [False]*(len(idx)+1)
    else:
        raise TypeError('idx interp is for idx of str type')

    return cutInds,ellipsis
        

def reshape(a, idx, group_ellipsis=False):
    """ 
    Reshape tensors
    
   
    idx: subscripts to split according to ','
         '...' means reshape(-1) if group ellipsis is Ture
               means leave untouched if group_ellipsis is False
    a:   ndarray to reshape
    
    Returns
    -------
    new_a:  reshaped ndarray

    """
    # idx0 = re.split(",", idx)
    # ellipse = [x == '...' for x in idx0]
    # splits  = [len(x) for x in idx0]
    # L = len(splits)
    # 
    # a_sh   = a.shape
    # new_sh = []

    # indL = 0
    # for i in range(L):
    #     indR = indL + splits[i]
    #     if ellipse[i]:
    #         if i==0 or i==L-1:  
    #             new_sh += [-1]
    #         else:
    #             indR = L-np.sum(splits[i+1:])
    #             new_sh += [s for s in a_sh[indL:indR]]
    #     else:
    #         new_sh += [np.prod(a_sh[indL:indR],dtype=np.int)]
    #     indL = indR  

    # if   isinstance(idx,str):   cutInds = idx_interp(idx)
    # elif isinstance(idx,int):   cutInds = [idx]
    # else:                       cutInds = idx   # assumes is list of integers


    cutInds,isEllipsis = idx_interp(idx,a.ndim)

    M = len(cutInds)
    
    a_sh   = a.shape
    new_sh = []

    if group_ellipsis or not isEllipsis[0]:   new_sh += [np.prod(a_sh[:cutInds[0]])]
    else:                                     new_sh += a_sh[:cutInds[0]]

    for i in range(0,M-1):
        c1,c2 = cutInds[i:i+2]
        if group_ellipsis or not isEllipsis[i+1]:        new_sh += [np.prod(a_sh[c1:c2],dtype=np.int)]
        else:                                            new_sh += a_sh[c1:c2]

    if group_ellipsis or not isEllipsis[-1]:  new_sh += [np.prod(a_sh[cutInds[-1]:])]
    else:                                     new_sh += a_sh[cutInds[-1]:]

    return a.reshape(new_sh)


# @profile
def svd(a, idx, DMAX=0, tol=None):
    """
    Thin Singular Value Decomposition

    idx : subscripts to split 
    a : ndarray
        matrix to do svd.
    DMAX: int
        maximal dim to keep.
     
    Returns
    -------
    u : ndarray
        left matrix
    s : ndarray
        singular value
    vt : ndarray
        right matrix
    dwt: float
        discarded wt
    """
    # idx0 = re.split(",", idx)
    # assert len(idx0) == 2
    # idx0[0].replace(" ", "")
    # nsplit = len(idx0[0]) 
    
    idx = idx_interp(idx, a.ndim)[0]
    nsplit = idx[0]

    a_shape = a.shape
    # print 'svd', a.shape, a_shape[:nsplit], a_shape[nsplit:]
    a = a.reshape(np.prod(a_shape[:nsplit]), -1)
    try:
        u, s, vt = scipy.linalg.svd(a, full_matrices=False)
    except(np.linalg.linalg.LinAlgError):
        print 'using gesvd'
        u, s, vt = scipy.linalg.svd(a, full_matrices=False,lapack_driver='gesvd')
    # u, s, vt = scipy.linalg.svd(a, full_matrices=False,lapack_driver='gesvd')
    
    # print 'tf svd', np.linalg.norm(a), np.linalg.norm(s)

    # sort_inds = np.argsort(np.abs(s[:]))[::-1]
    # sVals = s[sort_inds]
    # u  =  u[:,sort_inds]
    # vt = vt[sort_inds,:]

    sVals = s[:]
    if tol is not None:     s = sVals[abs(sVals) > tol*s[0]]
    else:                   pass
    # s = sVals[abs(sVals) > tol]
    # s     = sVals[abs(sVals/s[0]) > tol]   

    # print 'tf svd small', np.linalg.norm(a), np.linalg.norm(s)

    if len(s) == 0:  
        s = np.array([tol]) # np.array([0])
        # print 's is almost 0!'

    M = len(s)
    if DMAX > 0:
        M = min(DMAX, M)
    
    dwt = np.sum( sVals[M:]**2 )/(s[0]**2)

    # try:
    #     print 'svd', np.max(sVals[M:]), np.min(sVals[:M]), M, len(sVals)
    #     if np.min(sVals[:M]) > 1.0 - 1e-9:  
    #        print sVals
    #        print 'trunc', sVals[M:]
    # except:   # M = len(sVals), M = 0
    #     pass
 
    # u   = u[:,sort_inds[:M]].reshape(a_shape[:nsplit] + (-1,))
    u   = u[:,:M].reshape(a_shape[:nsplit] + (-1,))
    s   = s[:M]
    s   = s*np.linalg.norm(a)/np.linalg.norm(s)
    vt  = vt[:M,:].reshape((-1,) + a_shape[nsplit:])
    # vt  = vt[sort_inds[:M],:].reshape((-1,) + a_shape[nsplit:])

    # print u.shape, vt.shape, len(s)
    # print 'tf svd DMAX', np.linalg.norm(a), np.linalg.norm(s)

    # print 'svd', a_shape, u.shape, M, DMAX, dwt, np.sum( sVals[:]**2 )

    return u, s, vt, dwt


# @profile
def qr(a, idx):
    """
    QR decomposition

    idx : subscripts to split 
    a : ndarray
        matrix to do qr.
     
    Returns
    -------
    q : ndarray
        left matrix
    r : ndarray
        upper triangular matrix
    """
    
    idx = idx_interp(idx, a.ndim)[0]
    nsplit = idx[0]

    a_ = a.copy()
    a_shape = a.shape
    a = a.reshape(np.prod(a_shape[:nsplit]), -1)
    q,r = scipy.linalg.qr(a, mode='economic')

    q = q.reshape(a_shape[:nsplit] + (-1,))
    r = r.reshape((-1,) + a_shape[nsplit:])

    # print 'check qr', np.linalg.norm( np.tensordot(q,r,axes=(-1,0))-a_ )

    return q, r

# @profile
def lq(a, idx):
    """
    LQ decomposition

    idx : subscripts to split 
    a : ndarray
        matrix to do qr.
     
    Returns
    -------
    q : ndarray
        right matrix
    l : ndarray
        lower triangular matrix
    """
    
    idx = idx_interp(idx, a.ndim)[0]
    nsplit = idx[0]

    a_ = a.copy()
    a_shape = a.shape
    a = a.reshape(np.prod(a_shape[:nsplit]), -1)
    q,r = scipy.linalg.qr(np.conj(a.T), mode='economic')
    # q,r = scipy.linalg.qr(a.T, mode='economic')
    
    l = np.conj(r.T)
    q = np.conj(q.T)
    # l = r.T
    # q = q.T

    M = q.shape[0]
    l = l.reshape(a_shape[:nsplit] + (-1,))
    q = q.reshape((-1,) + a_shape[nsplit:])

    # print 'check lq', np.linalg.norm( np.tensordot(l,q,axes=(-1,0))-a_ )

    return l,q


# def dMult(order,o1,o2,div=False):
#     # order = 'DM' or 'MD'
#     # if 'DM':  o1 = col vec, o2 = mat, matshape[0] = len(vec)
#     # if 'MD':  o1 = mat, o2 = col vec, matshape[-1] = len(vec)
#     # all mats are m x (...) x n
#     # div:  catch the 1/0 error
# 
#     if o1 is None or o2 is None:
#         return None
#     else:    
#         if order == 'DM':       # col vec -> diag*mat
# 
#             if not (len(o1) == o2.shape[0]):
#                 print('dim mismatch',len(o1),o2.shape[0])
#                 exit()
# 
#             newMat = np.array(o2)               # need to make new copies of array
#             newVec = np.array(o1)
#             
#             if div:
#                 i = 0
#                 while i < len(o1) and newVec[i] < 1.e12:  i += 1
#                 newVec[i:] = 0
#                 # print i
#             
#             for i in range(len(o1)):
#                 newMat[i] = newMat[i]*newVec[i]
# 
#         elif order == 'MD':
#             newMat = np.array(o1)
#             newVec = np.array(o2)
#             
#             if not (len(o2) == o1.shape[-1]):
#                 print('dim mismatch',len(o2),o1.shape[-1])
#                 exit()
# 
#             if div:
#                 i = 0
#                 while i < len(o2) and newVec[i] < 1.e12:  i += 1
#                 newVec[i:] = 0
#                 # print i
#             
#             newMat = newMat.transpose()
#             for i in range(len(o2)):
#                 newMat[i] = newMat[i]*newVec[i]
#             newMat = newMat.transpose()
#             
#         else:
#             print('provide valid order')
#             exit()
#      
#         return newMat


# def dMult(order,o1,o2,div=False):
#     # order = 'DM' or 'MD'
#     # if 'DM':  o1 = col vec, o2 = mat, matshape[0] = len(vec)
#     # if 'MD':  o1 = mat, o2 = col vec, matshape[-1] = len(vec)
#     # all mats are m x (...) x n
#     # div:  catch the 1/0 error
# 
#     # print 'o1', o1.shape
#     # print 'o2', o2.shape
# 
#     if order == 'DM':       # col vec -> diag*mat
#         new_mat = np.tensordot(np.diag(o1),o2,axes=(1,0))
# 
#     elif order == 'MD':
#         new_mat = np.tensordot(o1,np.diag(o2),axes=(-1,0))
#    
#     # print new_mat.shape
#     return new_mat


def dMult(order,o1,o2):
    # order = 'DM' or 'MD'
    # if 'DM':  o1 = col vec, o2 = mat, matshape[0] = len(vec)
    # if 'MD':  o1 = mat, o2 = col vec, matshape[-1] = len(vec)
    # all mats are m x (...) x n

    if order == 'DM':
        new_mat = (o1*o2.T).T     # T just flips order of all axes

    elif order == 'MD':
        new_mat = o1*o2
   
    # print new_mat.shape
    return new_mat


def dot_diag(tens,diag,ax):

    if   ax == 0:                         return dMult('DM',diag,tens)
    elif ax == -1 or ax==tens.ndim-1:     return dMult('MD',tens,diag)
    else:
        ntens = np.moveaxis(tens,ax,-1)
        ntens = ntens*diag
        ntens = np.moveaxis(ntens,-1,ax)
        return ntens


def decompose_block(block, ns, direction, DMAX, svd_str=None, return_s=False):
    ''' ndarray block --> MPX via SVD (canonical form based on direction)
        return list of tensors (not MPX)
    '''
    temp = []
    s_list = []
    errs = []
    if direction == 0:
        if svd_str is None:  svd_str = 'ij,...'
        for i in range(ns-1):
            u,s,vt,dwt = svd(block,svd_str,DMAX)
            s_list.append(s)
            temp.append(u)
            errs.append(dwt)
            block = dMult('DM',s,vt)
        temp.append(dMult('DM',s,vt))
    elif direction == 1:
        if svd_str is None:  svd_str = '...,ij'
        for i in range(ns-1):
            u,s,vt,dwt = svd(block,svd_str,DMAX)
            s_list.append(s)
            temp.append(vt)
            errs.append(dwt)
            block = dMult('MD',u,s)
        temp.append(dMult('MD',u,s))

        temp = temp[::-1]
        errs = errs[::-1]
        s_list = s_list[::-1]
    else:
        print 'decompose_block: provide valide direction 0 or 1'
        exit()

    if return_s:  return temp, s_list
    else:         return temp



def decompose_block_GL(block, ns, DMAX, lamL=None, lamR=None, svd_str=None):
    ''' ndarray block --> gamma, lambdas via SVD
        return gammas, lambdas, errs (not MPX)
    '''

    if lamL is None:  lamL = np.ones(block.shape[0])
    if lamR is None:  lamR  = np.ones(block.shape[-1])
    
    lambdas = [lamL]   # len ns+1
    gammas  = []       # len ns
    errs = []          # len ns-1

    if svd_str is None:  svd_str = 'ij,...'
    for i in range(ns-1):
        u,s,vt,dwt = svd(block,svd_str,DMAX)
        lambdas.append(s)
        gammas.append( dMult('DM',1./lambdas[i],u) )
        errs.append(dwt)
        block = dMult('DM',s,vt)
    
    gammas.append(dMult('MD',vt,1./lamR))
    lambdas.append(lamR)

    # should probably include decomposition in the other direction..?

    return gammas,lambdas,errs


###############################
####   bond reordering   ######
###############################


def io2site(ns, nps=1):
    """
    obtain axes to reorder (multi-site) tensor from 'io' to 'site' ordering
    'io':    l x (u1 x u2 ... ) x (d1 x d2 ...) x r
    'site':  l x (u1 x d1) x (u2 x d2) ... x r

    ns:  number of sites the tensor represents
    nps: number of physical bond legs at each site (pointing up/down each) (normal mpo: nps=1)
    """

    return [0] + list(np.arange(1,ns*nps*2+1).reshape(2,ns,-1).transpose(1,0,2).reshape(-1)) + [2*ns*nps+1]


def site2io(ns, nps=1):
    """
    obtain axes to reorder (multi-site) tensor from 'site' to 'io' ordering
    'io':    l x (u1 x u2 ... ) x (d1 x d2 ...) x r
    'site':  l x (u1 x d1) x (u2 x d2) ... x r

    ns:  number of sites the tensor represents
    nps: number of physical bond legs at each site (pointing up/down each) (normal mpo: nps=1)
    """

    return [0] + list(np.arange(1,ns*nps*2+1).reshape(ns,2,-1).transpose(1,0,2).reshape(-1)) + [2*ns*nps+1]
