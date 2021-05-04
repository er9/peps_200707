import numpy as np
import scipy
# from scipy.sparse.linalg import lsmr
from scipy.sparse import linalg as LA
from scipy import optimize as Opt
# from scipy.optimize import minimize
import time


import tens_fcts as tf

import MPX as MPX
import PEPX
import PEPX_GL
import PEPS_env as ENV
import PEPS_GL_env_nolam as ENV_GL
import TLDM_GL_env_nolam as ENV_DM





# def meas_obs(peps,pepo,ind0=(0,0)):
#     # operator acts on sites ind0 : ind0+len(mpo)
#     L1,L2 = pepo.shape
#     ix,iy = ind0
# 
#     peps_ = peps.copy()
#     peps_[ix:ix+L1,iy:iy+L2] = dot(pepo,peps[ix:ix+L1,iy:iy+L2])
# 
#     expVal = PEPX.vdot(peps,peps_)
#     return expVal

def get_io_axT(in_legs,out_legs):
    return PEPX.get_io_axT(in_legs,out_legs,5)


def matricize(pepo_tens,iso_legs,iso_side):
    ''' peps tens is ndarray ludr(io)  
        iso_legs:  str or list of 'l','u','d','r':  legs to be on the grouped with io axes and placed on right
        iso_side = right: (not iso),(iso + io)
        iso_side = left:  (iso + io),(not iso)
        returns matrix, tens_shape (after reordering, but before reshaping), axT_inv
    '''
    return PEPX.matricize(pepo_tens,iso_legs,iso_side)
     
 
def unmatricize(peps_mat,tens_sh,axT_inv):
    '''  peps_mat is matrix (..., iso_legs + io)
         tens_sh:  shape of re-ordered tensor before reshaping
         return tens
    '''
    return PEPX.unmatricize(pepx_mat,tens_sh,axT_inv)
    


def opposite_leg(leg):   return PEPX.opposite_leg(leg)

def leg2ind(leg):        return PEPX.leg2ind(leg)


def QR_factor(pepx_tens,iso):
    ''' return Q, R, axT_inv '''
    return PEPX.QR_factor(pepx_tens,iso)

   
def LQ_factor(pepx_tens,iso):
    ''' return L, Q, axT_inv '''
    return PEPX.LQ_factor(pepx_tens,iso)


def QR_contract(Q,R,axT_inv):    
    tens = np.tensordot(Q,R,axes=(-1,0))
    return tens.transpose(axT_inv)
    

def LQ_contract(L,Q,axT_inv):
    tens = np.tensordot(L,Q,axes=(-1,0))
    return tens.transpose(axT_inv)





###########################################################
#### apply operators (trotter steps) to PEPX or PEPO ######
###########################################################

## change to just "block_update" because applies operator to pepo_list and just does svd ###
# @profile
def simple_update(gamma_list, lambda_list, block1, DMAX=10, num_io=1, direction=0, normalize=True):
    ''' algorithm:  contract all sites into block and then do svd compression
                    not the most efficient, but maybe numerically more stable?
        pepo_part:  list of pepo sites that we're operating on
        block1:     nparray mpo-like block (1 x ... x 1) applied to sites on 'i' vertical bond
    '''    

    if isinstance(block1,MPX.MPX):   # this needs to go first bc MPX is also instance of np.ndarray
        new_gams, new_lams, errs = PEPX_GL.mpo_update(gamma_list, lambda_list, block1, DMAX=DMAX, num_io=num_io,
                                                      direction=direction,normalize=normalize)#, chk_canon=True) 

    elif isinstance(block1, np.ndarray):
        new_gams, new_lams, errs = PEPX_GL.block_update(gamma_list, lambda_list, block1, DMAX=DMAX, num_io=num_io,
                                                        direction=direction,normalize=normalize)
    else:
        raise(TypeError), 'block1 should be np.ndarray (block) or MPX.MPX (mpo)'

    return new_gams, new_lams, errs


# def simple_block_mpo_update(peps_list,connect_list,trotter_block):
#     
#     L = len(peps_list)
#     trotter_mpo = tf.decompose(trotter_block, L, 0, -1, svd_str='iii,...')
# 
#     new_peps_list = PEPX.reduced_mpo_update(peps_list, connect_list, trotter_mpo)
#     return new_peps_list


#####################################################################
###        alternating least square methods                       ###
#####################################################################

def optimize_disentangler(g_env):
    ''' rerturn optimized disentangler g'''

    g_shape = g_env.shape
    sqdim = int(np.sqrt(np.prod(g_shape)))
    g_block = g_env.reshape(sqdim,sqdim)
    u,s,vt = np.linalg.svd(g_block)
    g = np.matmul(u,vt).reshape(g_shape)
    return g


def pos_pinv(sq_mat, rcond=1.0e-15, ensure_pos=False):

    u, s, vt = scipy.linalg.svd(sq_mat, full_matrices=False,lapack_driver='gesvd')
    s_ = s[abs(s) > rcond*s[0]]
    M  = len(s_)

    if ensure_pos:
    
        isH = np.allclose(sq_mat,np.conj(sq_mat.T))

        # if not isH:   print 'ensuring pos, not H'

        ### Reza's method:   sqrt(M^* M)
        pseudo_inv = np.dot(np.conj(vt[:M,:].T), tf.dMult('DM',1./s_,vt[:M,:]))

    else:
        # pseudo_inv = np.dot(u[:,:M],tf.dMult('DM',1./s_,vt[:M,:]))  ## old and wrong
        pseudo_inv = np.dot(np.conj(vt.T[:,:M]),tf.dMult('DM',1./s_,np.conj(u.T[:M,:])))

    return pseudo_inv 


def optimize_site(site_metric,site_grad,ensure_pos=False):
    ''' return one optimized site
       (site_metric -- reshape into sq matrix split along bonds for the two missing sites)
       (vector
    '''

    site_shape = site_metric.shape
    sqdim = int(np.sqrt(np.prod(site_metric.shape)))
    M_block = site_metric.transpose(0,2,4,6,1,3,5,7).reshape(sqdim,sqdim)   # lLiIoOrR -> (lior)x(LIOR)
    G_vec = site_grad.reshape(sqdim,-1)    # (lior)xd

    # print 'is hermitian?', np.allclose(M_block,np.conj(M_block.T))

    # M_inv = np.linalg.pinv(M_block, rcond=1.0e-8)
    M_inv = pos_pinv(M_block, rcond=1.0e-8, ensure_pos=False)
    site_ = np.matmul(M_inv,G_vec)

    # if np.linalg.norm(site_) > 5.0:
    #     print 'optimize site large norm', np.linalg.norm(site_), np.linalg.norm(G_vec),
    #     print np.linalg.norm(M_block), np.linalg.norm(M_inv)
    #     # exit()

    return site_.reshape(site_grad.shape)


def red_optimize_site(site_metric,site_grad,ensure_pos=False):
    ''' return one optimized site
       (site_metric -- reshape into sq matrix split along bonds for the two missing sites)
       (vector
    '''

    site_shape = site_metric.shape
    sqdim = int(np.sqrt(np.prod(site_metric.shape)))
    try:
        M_block = site_metric.transpose(0,2,1,3).reshape(sqdim,sqdim)   # qQxX -> (qx)x(QX)
    except(ValueError):
        M_block = site_metric.transpose(0,2,4,1,3,5).reshape(sqdim,sqdim)
    G_vec = site_grad.reshape(sqdim,-1)    # (qx)xd

    # print 'qr is hermitian?', np.allclose(M_block,np.conj(M_block.T))

    # M_inv = np.linalg.pinv(M_block, rcond=1.0e-8)
    M_inv = pos_pinv(M_block, rcond=1.0e-8, ensure_pos=False)
    site_ = np.matmul(M_inv,G_vec)

    return site_.reshape(site_grad.shape)


def optimize_site_2(site_metric,site_grad,do_lsq=True):

    db = site_grad.shape[-1] 
    
    metric_ = np.einsum('...,ij->...ij',site_metric,np.eye(db))
    sqdim = int(np.sqrt(np.prod(metric_.shape)))
    M_block = metric_.transpose(0,2,4,6,8,1,3,5,7,9).reshape(sqdim,sqdim)
    G_vec = site_grad.reshape(-1)

    if do_lsq:
        site_, info_stop, it_num, diff = LA.lsmr(M_block,G_vec)[:4]
    else:
        M_inv = pos_pinv(M_block, rcond=1.0e-8)
        site_ = np.matmul(M_inv,G_vec)

    return site_.reshape(site_grad.shape)


## tested -- not stable as dimension gets larger ##
def optimize_site_constrained(site_metric,site_grad,do_reduced=False):

    db = site_grad.shape[-1] 
    
    metric_ = np.einsum('...,ij->...ij',site_metric,np.eye(db))
    sqdim = int(np.sqrt(np.prod(metric_.shape)))
    G_vec = site_grad.reshape(-1)
    if do_reduced:
        try:                  # 1-leg QR
            M_block = metric_.transpose(0,2,4,1,3,5).reshape(sqdim,sqdim)
        except(ValueError):   # 2-leg QR 
            M_block = metric_.transpose(0,2,4,6,1,3,5,7).reshape(sqdim,sqdim)
    else:
        M_block = metric_.transpose(0,2,4,6,8,1,3,5,7,9).reshape(sqdim,sqdim)

    init_guess = LA.lsmr(M_block,G_vec)[0]

    def opt_norm(site_vec,norm=1.):
        opt_N = np.dot( np.conj(site_vec),np.dot(M_block,site_vec) )
        return opt_N - norm

    def fct_to_min(site_vec):
        return np.dot(np.conj(site_vec).T, np.dot(M_block,site_vec)) - np.dot(G_vec,site_vec)
        # return np.dot(G_vec,site_vec)*-1

    cons = ({'type':'eq','fun':opt_norm},) #{'type':'ineq','fun':lambda x: x})
    # bnds = Opt.Bounds( np.zeros(sqdim), np.ones(sqdim)*np.inf )
    result = Opt.minimize(fct_to_min, (init_guess,), constraints=cons , tol=1.0e-3, method='SLSQP')  
    # result = Opt.minimize(fct_to_min, init_guess, constraints=cons , method='trust-constr') #'SLSQP')  
    if not result.success:
        print result.x
        print result.message, result.nit
        raise RuntimeError('constrained optimization failed')

    return result.x.reshape(site_grad.shape)


## not tested ##
def optimize_site_constrained_2(site_metric,site_grad,do_reduced=False,rcond=1.0e-8,tol=1.0e-3):
    ''' site_metric:  o opt | o opt
        site_grad:    o opt | x dt 
    '''

    db = site_grad.shape[-1] 
    
    metric_ = np.einsum('...,ij->...ij',site_metric,np.eye(db))
    sqdim = int(np.sqrt(np.prod(metric_.shape)))
    G_vec = site_grad.reshape(-1)
    if do_reduced:
        try:                  # 1-leg QR
            M_block = metric_.transpose(0,2,4,1,3,5).reshape(sqdim,sqdim)
        except(ValueError):   # 2-leg QR 
            M_block = metric_.transpose(0,2,4,6,1,3,5,7).reshape(sqdim,sqdim)
    else:
        M_block = metric_.transpose(0,2,4,6,8,1,3,5,7,9).reshape(sqdim,sqdim)

    ### obtain M = X.T * X
    # M_sqrt = np.linalg.cholesky(M_block)
    u, s, vt = scipy.linalg.svd(M_block, full_matrices=False,lapack_driver='gesvd')
    s_ = s[abs(s) > rcond*s[0]]
    M  = len(s_)

    # assume postive -- u = vt.T
    is_pos = np.allclose(u,np.conj(vt.T))
    if not is_pos:   pass # print 'warning, M_block is not pos'
    ### seems like generally M is not definite positive

    M_sq    = tf.dMult('DM',np.sqrt(s_),vt[:M,:])
    Minv_sq = tf.dMult('DM',np.sqrt(1./s_),vt[:M,:])   # => (M^-1) = A.T A;  also Minv_sq.T --> inv(M_sq)
    
    GM_vec = np.dot(Minv_sq,G_vec)

    init_guess = np.zeros(G_vec.shape)

    def fct_to_min(diff_vec):
        return np.linalg.norm(diff_vec) + np.dot(np.conj(diff_vec),GM_vec) + np.dot(np.conj(GM_vec),diff_vec)

    cons = ({'type':'ineq','fun':lambda x: tol-np.linalg.norm(x)},)    # ||c|| < epsilon --> eps-||c|| < 0 
    result = Opt.minimize(fct_to_min, (init_guess,), constraints=cons , method='SLSQP')  
    # result = Opt.minimize(fct_to_min, init_guess, constraints=cons , method='trust-constr') #'SLSQP')  
    if not result.success:
        print result.x
        print result.message, result.nit
        raise RuntimeError('constrained optimization failed')

    opt_diff = result.x
    M_sq_x = opt_diff + GM_vec
    opt_x = np.dot( Minv_sq.T, M_sq_x )

    return opt_x.reshape(site_grad.shape)


## not tested ##
def optimize_site_constrained_3(site_metric,site_grad,do_reduced=False):

    db = site_grad.shape[-1] 
    
    metric_ = np.einsum('...,ij->...ij',site_metric,np.eye(db))
    sqdim = int(np.sqrt(np.prod(metric_.shape)))
    G_vec = site_grad.reshape(-1)
    if do_reduced:
        try:                  # 1-leg QR
            M_block = metric_.transpose(0,2,4,1,3,5).reshape(sqdim,sqdim)
        except(ValueError):   # 2-leg QR 
            M_block = metric_.transpose(0,2,4,6,1,3,5,7).reshape(sqdim,sqdim)
    else:
        M_block = metric_.transpose(0,2,4,6,8,1,3,5,7,9).reshape(sqdim,sqdim)

    init_guess = LA.lsmr(M_block,G_vec)[0]

    def fct_to_min(site_vec):
        return np.dot(np.conj(site_vec).T, np.dot(M_block,site_vec)) - np.dot(G_vec,site_vec)
        # return -1*np.dot(G_vec.T,site_vec) - np.dot(site_vec.T, G_vec)
        ### i don't understand why we need the x.T M x term when we're setting a constraint on that...
        ### but otherwise it's just not stable?

    def fct_jac(x_vec):
        return np.dot(np.conj(x_vec).T, M_block) - G_vec
 
    def fct_hess(x_vec):
        # return np.zeros(M_block.shape)
        return M_block

    def opt_norm(site_vec):
        opt_N = np.dot( np.conj(site_vec),np.dot(M_block,site_vec) )
        return opt_N

    def constr_jac(x_vec):
        return np.dot( np.conj(x_vec), M_block )

    def constr_hess(x_vec, v_vec):
        return M_block

    tol = 1.0e-3
    nonlinear_constraint = Opt.NonlinearConstraint( opt_norm, 1-tol, 1+tol, jac=constr_jac, hess=constr_hess )
    bnds = None  # Opt.Bounds( np.zeros(sqdim), np.ones(sqdim)*np.inf )
    result = Opt.minimize(fct_to_min, init_guess, method='trust-constr', jac=fct_jac, hess=fct_hess,
                          constraints=nonlinear_constraint, bounds=bnds)  
    if not result.success:
        print result.x
        print result.message, result.nit
        raise RuntimeError('constrained optimization failed')

    return result.x.reshape(site_grad.shape)



###############################################
#####   alsq update intermediate method  ######
###############################################

# @profile
def alsq_block_update(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                      init_guess=None, ensure_pos=False, normalize=False, regularize=True, build_env=False,
                      qr_reduce=False, flatten=False, alsq_2site=False, max_it=500,sub_max_it=50,conv_tol=1.e-8,
                      sub_conv_tol=1.0e-8):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        xy_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over
    '''

    max_it = 100
    sub_max_it = 5
    ## 200718
    conv_tol = 1.e-6
    sub_conv_tol = 1.e-6   
    ## 200707
    # conv_tol = 1.e-8
    # sub_conv_tol = 1.e-8

    if qr_reduce:

        ### QR methods ###
        if sub_peps_gl.shape == (2,2):
            # print 'doing 2x2 reduced alsq'
            return alsq_redenv_2x2(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX,XMAX,site_idx,
                                   init_guess, ensure_pos, normalize, flatten, max_it, sub_max_it, conv_tol, sub_conv_tol)
        else:
            # print 'doing 2-site reduced alsq'
            return alsq_redenv(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX,XMAX,site_idx,
                               init_guess, ensure_pos, normalize, flatten, max_it, conv_tol)

    ### no QR reduce, but do 2-site optimization ###
    if sub_peps_gl.shape == (2,2) and alsq_2site:
        # print 'doing alsq 2site'
        return alsq_block_update_2x2(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX,XMAX,site_idx,
                                     init_guess, ensure_pos, normalize, regularize, build_env,
                                     qr_reduce, flatten, max_it, sub_max_it, conv_tol, sub_conv_tol)

    # print 'doing gen update'
    return alsq_gen(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX=DMAX,XMAX=XMAX,site_idx=site_idx,
                      init_guess=init_guess, ensure_pos=ensure_pos, normalize=normalize, regularize=regularize,
                      build_env=build_env, qr_reduce=qr_reduce, flatten=flatten, max_it=max_it, conv_tol=conv_tol)



###############################################
#####   general full tensor alsq update  ######
###############################################

def alsq_gen(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
             init_guess=None, ensure_pos=False, normalize=False, regularize=True, build_env=False,
             qr_reduce=False, flatten=False, max_it=500,conv_tol=1.0e-8):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        xy_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over
    '''

    L1,L2 = sub_peps_gl.shape
    num_io = sub_peps_gl[0,0].ndim-4


    ind0 = (xy_list[0][0],xy_list[1][0])
    gam_list, lam_list, axT_invs = PEPX_GL.get_sites(sub_peps_gl,ind0,connect_list)

    # apply trotter and then svd
    gam_dt, lam_dt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=-1,num_io=num_io,direction=0,
                                         normalize=normalize)
    peps_dt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_dt,lam_dt,axT_invs) 
    # print 'peps dt', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)], [m.shape for m in trotter_block]

    if flatten:
        dbs = peps_dt_gl.phys_bonds
        peps_dt_gl  = PEPX_GL.flatten(peps_dt_gl)
        # peps_opt_gl = PEPX_GL.flatten(peps_opt_gl)
        # print 'flattened', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)]

    if build_env:     norm_env = ENV_GL.build_env(env_list,ensure_pos=ensure_pos)

    if normalize:
        if build_env:        dt_norm = ENV_GL.embed_sites_norm_env(peps_dt_gl, norm_env)   # env is without lambdas
        else:                dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
        # print 'dt norm', dt_norm
        peps_dt_gl = PEPX_GL.mul(1./dt_norm, peps_dt_gl)


    if isinstance(init_guess,np.ndarray): 
         # init guess -- increase bond dimension using svd init guess
         leg_ind = PEPX.leg2ind(connect_list[0])
         if sub_peps_gl[0,0].shape[leg_ind] < DMAX:
             # peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)
             gam_opt, lam_opt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=DMAX,num_io=num_io,
                                                    direction=0,normalize=normalize)
             peps_opt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_opt,lam_opt,axT_invs) 
         else:
             peps_opt_gl = init_guess

         if flatten:
             peps_opt_gl  = PEPX_GL.flatten(peps_opt_gl)

    elif init_guess in ['rand','random']:
         leg_ind_o = [PEPX_GL.leg2ind(c_leg) for c_leg in connect_list]+[None]
         leg_ind_i = [None]+[PEPX_GL.leg2ind(PEPX_GL.opposite_leg(c_leg)) for c_leg in connect_list]
         shape_array = np.empty((L1,L2),dtype=tuple)
         for i in range(len(peps_list)):
             ind = (xs[i],ys[i])
             rand_shape = list(peps_list[i].shape)
             if leg_ind_o[i] is not None:     rand_shape[leg_ind_o[i]] = DMAX
             if leg_ind_i[i] is not None:     rand_shape[leg_ind_i[i]] = DMAX
             shape_array[ind] = tuple(rand_shape)

         raise(NotImplementedError), 'random init state for alsq not implemented'

    else:     # init_guess is None or init_guess in ['su','SU','svd']:
         # peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)
         gam_opt, lam_opt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=DMAX,num_io=num_io,
                                                direction=0,normalize=normalize)
         peps_opt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_opt,lam_opt,axT_invs) 

         if flatten:
             peps_opt_gl  = PEPX_GL.flatten(peps_opt_gl)

         # print 'init peps opt', [m.shape for xx,m in np.ndenumerate(peps_dt_gl)],[m.shape for xx,m in np.ndenumerate(peps_opt_gl)], DMAX
         # bL, bI, bO, bR = env_list
         # print [m.shape for m in bL],[m.shape for m in bI],[m.shape for m in bO],[m.shape for m in bR]


    ## define order of sites in ALSQ optimization
    ## original (2,2)
    # site_idx = [(0,0),(0,1),(1,1),(1,0)]
    # indxs = [site_idx]

    # # if qr_reduce:
    # temp_xs = [m[0] for m in site_idx]
    # temp_ys = [m[1] for m in site_idx]
    # loop_conn   = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)
    # loop_conn_r = PEPX.get_inds_conn([temp_xs[::-1],temp_ys[::-1]],wrap_inds=True)
    # conns = [loop_conn]

    if site_idx is not None:
        temp_xs = [m[0] for m in site_idx]
        temp_ys = [m[1] for m in site_idx]
        loop_conn   = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)
        indxs = [site_idx]
        conns = [loop_conn]
    else:
        xs,ys = xy_list

        if (L1,L2) == (2,2):
            if len(xs) == 3:
                # q1 = (xs[0],ys[0])
                # q2 = (xs[2],ys[2])
                qm = ((xs[1]+1)%2,(ys[1]+1)%2)
                # q_inds = [(q1,qm),(qm,q2)]

                site_idx = [(xs[i],ys[i]) for i in range(len(xs))]
                indxs = [site_idx+[qm],[qm]+site_idx[::-1]]
                conns = []
                for indx in indxs:
                    temp_xs = [m[0] for m in indx]
                    temp_ys = [m[1] for m in indx]
                    # loop_conn = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=False)
                    # conns += [loop_conn + PEPX_GL.opposite_leg(loop_conn[-1])]
                    loop_conn = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)
                    conns += [loop_conn]

            elif len(xs) == 4:
                q1 = (xs[0] ,ys[0] )
                q2 = (xs[-1],ys[-1])
                q_inds = [(q1,q2)]

                site_idx = [(xs[i],ys[i]) for i in range(len(xs))]
                indxs = []
                conns = []
                for i in range(4):
                    indx = site_idx[i:] + site_idx[:i]
                    temp_xs = [m[0] for m in indx]
                    temp_ys = [m[1] for m in indx]
                    loop_conn = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=False)
                    indxs += [indx]
                    conns += [loop_conn + PEPX_GL.opposite_leg(loop_conn[-1])]

        else:
            site_idx = [(xs[i],ys[i]) for i in range(len(xs))]
            temp_xs = [m[0] for m in site_idx]
            temp_ys = [m[1] for m in site_idx]
            loop_conn   = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)
            indxs = [site_idx]
            conns = [loop_conn]


    # print 'alsq gen', indxs, conns
    # print [(idx,m.shape) for idx,m in np.ndenumerate(peps_opt_gl)]
    # print [(idx,m.shape) for idx,m in np.ndenumerate(peps_dt_gl)]
    # print DMAX, XMAX

    peps_opt_gl = alsq_gen_update((L1,L2), indxs, conns, peps_dt_gl, peps_opt_gl, env_list, env_list, normalize=normalize,
                            ensure_pos=ensure_pos, build_env=build_env, qr_reduce=qr_reduce, max_it=max_it, conv_tol=conv_tol )

    if flatten:
        peps_opt_gl = PEPX_GL.unflatten(peps_opt_gl,dbs)

    return peps_opt_gl
 

# @profile
def alsq_gen_update(Ls, indxs, conns, peps_dt_gl, peps_init_gl ,env_list_m, env_list_g, normalize=False, 
                    regularize=True,ensure_pos=False,build_env=False,qr_reduce=False, max_it=500,conv_tol=1.e-8):
    ''' Ls = (L1,L2)
        indxs:  [list of [ list of site idxs indicating order of optimization ]]
        conns:  [list of [ list of loop conns indicating in/out bonds at site idx above ]]
        peps_dt_gl, peps_opt_gl:  pepx_gl objects (exact dt, init guess to be optimized)
        env_list:  [bL,bI,bO,bR], _m : for metric (opt/opt) , _g : for grad (opt/dt)
    '''

    L1,L2 = Ls
    peps_opt_gl = peps_init_gl.copy()
    old_peps_opt_gl = peps_opt_gl

    if build_env:
        norm_env_m = ENV_GL.build_env(env_list_m,ensure_pos=ensure_pos)
        norm_env_g = ENV_GL.build_env(env_list_g,ensure_pos=ensure_pos)

    not_converged = True
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0

    break_flag = False
    reuse_c = True
    oldc_x, oldc_xx = [None, None]   # remembering old corners to skip extra computations
    # while (not_converged or it < 3) and it < max_it:
    while (not_converged) and it < max_it:     # 200726 change

        for xi in range(len(indxs)):
        
            site_list = indxs[xi]
            site_conn = conns[xi]

            for xj in range(len(site_list)):

                idx = site_list[xj]
                iso_leg = site_conn[xj]

                if qr_reduce and (xj == 0 or xj == len(site_list)-1):

                    env_xx,qu_idx,qd_idx,axT_inv,oldc_xx = \
                       ENV_GL.red_embed_sites_xx(np.conj(peps_opt_gl),peps_opt_gl,env_list_m,idx,iso_leg,
                                                     old_corners=oldc_xx,return_corners=True)
                    env_x,oldc_x = \
                       ENV_GL.red_embed_sites_xo(np.conj(peps_opt_gl),peps_dt_gl,env_list_g,idx,iso_leg,qu_idx,qd_idx,
                                                     old_corners=oldc_x,return_corners=True)

                    r_opt = red_optimize_site(env_xx,env_x,ensure_pos=ensure_pos)   # includes lams
                    # r_opt = optimize_site_constrained(env_xx,env_x,do_reduced=qr_reduce)

                    site_opt = PEPX.QR_contract(qd_idx,r_opt,axT_inv,d_end=True)

                else:
                    # print 'opt',[ma.shape for xx, ma in np.ndenumerate(peps_opt_gl)]
                    # print 'dt', [ma.shape for xx, ma in np.ndenumerate(peps_dt_gl)]
                    env_x , oldc_x  = ENV_GL.embed_sites_xo(np.conj(peps_opt_gl),peps_dt_gl ,env_list_g, idx,
                                                            old_corners=oldc_x, return_corners=True)
                    env_xx, oldc_xx = ENV_GL.embed_sites_xx(np.conj(peps_opt_gl),peps_opt_gl,env_list_m,idx,
                                                             old_corners=oldc_xx,return_corners=True)

                    site_opt = optimize_site(env_xx,env_x,ensure_pos=ensure_pos)   # includes lams
                    # site_opt = optimize_site_2(env_xx,env_x,do_lsq=True)   # includes lams
                    # site_opt = optimize_site_constrained(env_xx,env_x)   # includes lams
                    # site_opt = optimize_site_constrained_2(env_xx,env_x)   # includes lams
                    # site_opt = optimize_site_constrained_3(env_xx,env_x)   # includes lams

                # print np.linalg.norm(site_opt), np.linalg.norm( PEPX_GL.get_site(peps_dt_gl,idx) )


                ## only change gamma (keep lambda from dt svd)
                # peps_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,peps_opt_gl.lambdas[idx])

                if not reuse_c:   oldc_x, oldc_xx = None,None

                ## do svd on site_opt + next site to keep it close to GL form
                peps_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,peps_opt_gl.lambdas[idx])
                gs, ls, axTs = PEPX_GL.get_sites(peps_opt_gl,idx,iso_leg)
                gs_, ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,normalize=False)
                peps_opt_gl = PEPX_GL.set_sites(peps_opt_gl,idx,iso_leg,gs_,ls_,axTs)

                # if normalize:
                #     opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list, old_corners=oldc_xx)
                #     gs_,ls_ = PEPX_GL.mul_GL_list(gs_,ls_,1./opt_norm)
                #     peps_opt_gl = PEPX_GL.set_sites(peps_opt_gl,idx,iso_leg,gs_,ls_,axTs)
                
                 
                # denote which site is modified (old corner cannot be reused)
                # only is affected when above setting of sites is not the same direction as next optimization step
                # eg. when loop direction is changed as in QR method
                if reuse_c and oldc_xx is not None:
                    mod_x,mod_y = PEPX.get_conn_inds(iso_leg,idx)
                    mod_ind = (mod_x[-1], mod_y[-1])
                    if  mod_ind == (0,0):
                        oldc_xx[0] = None
                        oldc_x [0] = None
                    elif mod_ind == (0,1):
                        oldc_xx[1] = None
                        oldc_x [1] = None
                    elif mod_ind == (1,0):
                        oldc_xx[2] = None
                        oldc_x [2] = None
                    elif mod_ind == (1,1):
                        oldc_xx[3] = None
                        oldc_x [3] = None            


                if np.any( [np.linalg.norm(peps_opt_gl.lambdas[xx]) > 10 for xx in np.ndindex(L1,L2,4)] ):
                    print 'large lambda', idx, iso_leg, [np.linalg.norm(peps_opt_gl.lambdas[xx]) 
                                                            for xx in np.ndindex(L1,L2,4)]
                    print 'large lambda', idx, iso_leg, [np.linalg.norm(peps_dt_gl.lambdas[xx]) 
                                                            for xx in np.ndindex(L1,L2,4)]
                    print [np.linalg.norm(m) for idx, m in np.ndenumerate(peps_opt_gl)]
                    opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list, old_corners=None)
                    dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list, old_corners=None)
                    print opt_norm, dt_norm
                    raise RuntimeWarning('large lambda, '+str(idx))

            ############

            # print 'peps opt 1', [m.shape for xx,m in np.ndenumerate(peps_dt_gl)],[m.shape for xx,m in np.ndenumerate(peps_opt_gl)], DMAX

            if regularize:    ## blows up if no regularization is done
                peps_opt_gl = PEPX_GL.regularize(peps_opt_gl)
                oldc_x, oldc_xx = None,None
                # print 'regularize', [np.linalg.norm(m) for m in gam_opt], lam_opt

            # print 'peps opt 2', [m.shape for xx,m in np.ndenumerate(peps_dt_gl)],[m.shape for xx,m in np.ndenumerate(peps_opt_gl)], DMAX

            if build_env:
                opt_norm = ENV_GL.embed_sites_norm_env(peps_opt_gl, norm_env_m)
                fidelity = ENV_GL.embed_sites_ovlp_env(np.conj(peps_dt_gl),peps_opt_gl,norm_env_g)/ opt_norm
            else:
                opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list_m, old_corners=oldc_xx)
	        fidelity = ENV_GL.embed_sites_ovlp(np.conj(peps_opt_gl),peps_dt_gl,env_list_g,old_corners=oldc_x) \
                                    /opt_norm

                if np.isnan(opt_norm):
                    print 'alsq gen update nan opt norm!', opt_norm

                    print 'dt norm'
                    print ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
                    print 'init norm'
                    print ENV_GL.embed_sites_norm(peps_init_gl, env_list)
                   
                    # print 'init norm'
                    # if flatten:    sub_peps_gl = PEPX_GL.flatten(sub_peps_gl)
                    # print ENV_GL.embed_sites_norm(sub_peps_gl, env_list)

                    raise RuntimeError('optimal norm is nan')

            not_converged = np.abs(fidelity - prev_fidelity) > conv_tol #1.0e-6  #1.0e-8  #1.0e-10
            fid_diff = fidelity-prev_fidelity

            # print 'fidelity', it, fidelity, opt_norm, fid_diff

            if np.imag(fidelity) > np.real(fidelity):
                print fidelity, opt_norm
                print 'imag opt', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
                print 'real opt', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
                print 'imag dt',  [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
                print 'real dt',  [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
                if build_env:  print np.linalg.norm(norm_env)
                raise RuntimeWarning('imag fidelity or norm')


            if (prev_fidelity > fidelity+1.0e-12):
                peps_opt_gl = old_peps_opt_gl
                oldc_x, oldc_xx = None, None
                print 'fidelity went down or alsq diff went up, reset opt gl'
                break_flag = True
                break

            # prev_alsq_diff = alsq_diff
            prev_fidelity     = fidelity
            old_peps_opt_gl   = peps_opt_gl

        if break_flag:   break

        it += 1

    if not_converged:  print 'not converged', it, fid_diff

    if normalize:
        peps_opt_gl = PEPX_GL.mul(1./opt_norm, peps_opt_gl)

    return peps_opt_gl



###############################################
##### reduced update methods for 2-sites ######
###############################################

def qr_list_to_dict(gam_qr,lam_qr,idxs):
    gam_dict = {}
    for i in range(len(idxs)):
        if i == len(idxs)-1:
            gam_dict[idxs[i]] = np.swapaxes(gam_qr[i], 0,-1)
        else:
            gam_dict[idxs[i]] = gam_qr[i]
    return gam_dict


def dict_to_qr_list(gl_dict,idxs):
    gam_list = []
    for i in range(len(idxs)):
        gam = gl_dict[idxs[i]]
        if i == len(idxs)-1:   gam_list += [np.swapaxes(gam,0,-1)]
        else:                  gam_list += [gam]
    return gam_list


def alsq_redenv(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                init_guess=None, ensure_pos=False, normalize=False, flatten=False, max_it=500, conv_tol=1.0e-8):

    ''' optimize each tens in peps_list, given environment
    '''

    L1,L2 = sub_peps_gl.shape
    assert ((L1,L2) == (1,2) or (L1,L2) == (2,1)), 'qr + env only can be done for 2x1 or 1x2 sub_peps'
    num_io = sub_peps_gl[0,0].ndim-4
    # print 'redenv num io', num_io
    # print [m.shape for idx,m in np.ndenumerate(sub_peps_gl)]

    xs,ys = xy_list
    idx_list = [(xs[i],ys[i]) for i in range(len(xs))]

    ind0 = (xs[0],ys[0])
    gam_list, lam_list, axT_invs = PEPX_GL.get_sites(sub_peps_gl,ind0,connect_list)
    # print [m.shape for m in gam_list]

    ## do QR decomposition before doing SU
    gam_list_qr, lam_list_qr, q0, qL = PEPX_GL.full_to_qr_GL_list(gam_list,lam_list,num_io)
    # print [m.shape for m in gam_list_qr], q0.shape, qL.shape
    lam0,lamL = lam_list[0],lam_list[-1]
        # gam list_qr:  first el looks like R (idq), last el is L (qdo)
        # lam list_qr:  first and last el's are just 1's (need to check that these aren't modified anywhere...)
        # q0, qL: both are Q's from QR

    # apply trotter and then svd
    gam_dt_qr, lam_dt_qr, errs = simple_update(gam_list_qr,lam_list_qr,trotter_block,DMAX=-1,direction=0,
                                               num_io=num_io,normalize=normalize)

    # gam_dt, lam_dt = PEPX_GL.qr_to_full_GL_list(gam_dt_qr,lam_dt_qr,q0,qL,lam0,lamL)
    # peps_dt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_dt,lam_dt,axT_invs) 
    # # print 'peps dt', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)], [m.shape for m in trotter_block]

    ####
    # gam_dt_T, lam_dt_T, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=-1,num_io=num_io,direction=0,
    #                                            normalize=normalize)
    # peps_dt_gl_T = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_dt_T, lam_dt_T, axT_invs)

    if flatten:
        dbs = sub_peps_gl.phys_bonds
        # sub_peps_gl = PEPX_GL.flatten(sub_peps_gl)
        # peps_dt_gl  = PEPX_GL.flatten(peps_dt_gl)
        # print [m.shape for m in gam_dt_qr]
        gam_dt_qr = [tf.reshape(m,'...,ii,i') for m in gam_dt_qr]
        # print [m.shape for m in gam_dt_qr]
        # peps_opt_gl = PEPX_GL.flatten(peps_opt_gl)
        # print 'flattened', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)]

    # ensure_pos = False #True
    norm_env = ENV_GL.build_env_qr(env_list,[q0,qL],[q0,qL],ensure_pos=ensure_pos)

    # if (L1,L2) == (1,2):
    #     norm_ = ENV_GL.build_env(env_list)
    #     # print norm_.shape, q0.shape, qL.shape, idx_list
    #     norm_ = np.einsum('lLiIjJoOpPrR,liox,LIOX->xXoOpPrR',norm_,np.conj(q0),q0)
    #     norm_ = np.einsum('rioq,RIOQ,lLiIoOrR->lLqQ',np.conj(qL),qL,norm_)
    # elif (L1,L2) == (2,1):
    #     norm_ = ENV_GL.build_env(env_list)
    #     norm_ = np.einsum('lLmMiIoOrRsS,ilrx,ILRX->mMxXoOsS',norm_,np.conj(q0),q0)
    #     norm_ = np.einsum('olrq,OLRQ,lLiIoOrR->iIqQ',np.conj(qL),qL,norm_)

    # print 'env diff', np.linalg.norm( norm_ - norm_env)

    if normalize:
        # dt_norm_T = dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl_T, env_list)

        # gam_dt, lam_dt = PEPX_GL.qr_to_full_GL_list(gam_dt_qr,lam_dt_qr,q0,qL,lam0,lamL)
        # peps_dt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_dt,lam_dt,axT_invs) 
        # dt_norm_p = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)

        gam_dict_dt = {}
        for i in range(len(gam_dt_qr)):
            # tens = tf.dMult('DM',lam_dt_qr[i], tf.dMult('MD',gam_dt_qr[i],lam_dt_qr[i+1]))
            if i == len(gam_dt_qr)-1:   gam_dict_dt[idx_list[i]] = np.swapaxes( gam_dt_qr[i], -1, 0 )
            else:                       gam_dict_dt[idx_list[i]] = gam_dt_qr[i]

        dt_norm = ENV_GL.red_embed_sites_norm_env((L1,L2),gam_dict_dt,lam_dt_qr[1],norm_env)  # env is without lambdas
        gam_dt_qr, lam_dt_qr = PEPX_GL.mul_GL_list(gam_dt_qr,lam_dt_qr,1./dt_norm)

        # dt_norm2 = ENV_GL.red_embed_sites_norm_env((L1,L2),gam_dict_dt,lam_dt_qr[1],norm_)

        # print 'dt norm', dt_norm_T, dt_norm_p, dt_norm, dt_norm2

        # metric1 = ENV_GL.red_embed_sites_xx_env((L1,L2), gam_dict_dt, lam_dt_qr[1], gam_dict_dt, lam_dt_qr[1],
        #                                          norm_env, (0,0))
        # metric2 = ENV_GL.embed_sites_xx(np.conj(peps_dt_gl),peps_dt_gl, env_list, (0,0))
        # print metric2.shape, [m.shape for idx,m in np.ndenumerate(peps_dt_gl_T)]
        # metric2 = np.einsum('lLiIoOrR,lioq,LIOQ->qQrR',metric2,np.conj(q0),q0)
        # print 'metric diff', np.linalg.norm(metric1-metric2), metric1.shape, metric2.shape
        # peps01 = PEPX_GL.get_site(peps_dt_gl,(0,1),no_lam=[0])
        # qr01   = np.einsum('rioq,qdl->liord',qL,gam_dict_dt[0,1])
        # print '01 site diff', np.linalg.norm(peps01-qr01)
        # peps00 = PEPX_GL.get_site(peps_dt_gl,(0,0),no_lam=[3])
        # qr00   = np.einsum('lioq,qdr->liord',q0,gam_dict_dt[0,0])
        # print '00 site diff', np.linalg.norm(peps00-qr00)

        # peps01 = PEPX_GL.get_site(peps_dt_gl,(0,1),no_lam=[])
        # qr01   = np.einsum('rioq,qdl->liord',qL,tf.dMult('MD',gam_dict_dt[0,1],lam_dt_qr[1]))
        # print '01 site diff', np.linalg.norm(peps01-qr01)
        # peps00 = PEPX_GL.get_site(peps_dt_gl,(0,0),no_lam=[])
        # qr00   = np.einsum('lioq,qdr->liord',q0,tf.dMult('MD',gam_dict_dt[0,0],lam_dt_qr[1]))
        # print '00 site diff', np.linalg.norm(peps00-qr00)

        # # r00 = tf.dMult('MD',gam_dt_qr[0],lam_dt_qr[1])
        # print [m.shape for m in lam_dt_qr], [gam_dict_dt[k].shape for k in np.ndindex(L1,L2)]
        # r00 = tf.dMult('MD',gam_dict_dt[0,0],lam_dt_qr[1])
        # norm3 = np.einsum('qQrR,qdr,QdR->',metric2,np.conj(r00),r00)
        # print 'dt norm', norm3, lam_dt_qr[0], lam_dt_qr[-1]

        # # print 'dt lam', lam_dt_qr
        # peps00 = PEPX_GL.get_site(peps_dt_gl,(0,0),no_lam=[])
        # peps01 = PEPX_GL.get_site(peps_dt_gl,(0,1),no_lam=[])
        # metric4 = ENV_GL.embed_sites_xx(np.conj(peps_dt_gl),peps_dt_gl, env_list, (0,0))
        # norm4 = np.einsum('lLiIoOrR,liord,LIORd->',metric4,np.conj(peps00),peps00)
        # print 'dt norm', np.sqrt(norm4)
        # metric4 = ENV_GL.embed_sites_xx(np.conj(peps_dt_gl),peps_dt_gl, env_list, (0,1))
        # norm4 = np.einsum('lLiIoOrR,liord,LIORd->',metric4,np.conj(peps01),peps01)
        # print 'dt norm', np.sqrt(norm4)  

        # grad4 = ENV_GL.embed_sites_xo(np.conj(peps_dt_gl),peps_dt_gl, env_list, (0,0))
        # norm4 = np.einsum('liord,liord->',np.conj(peps00),grad4)
        # print 'dt grad norm', np.sqrt(norm4)
        # grad4 = ENV_GL.embed_sites_xo(np.conj(peps_dt_gl),peps_dt_gl, env_list, (0,1))
        # norm4 = np.einsum('liord,liord->',np.conj(peps01),grad4)
        # print 'dt grad norm', np.sqrt(norm4)

        # dt_norm3 = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
        # print 'dt norm peps', dt_norm3

    # exit()


    if isinstance(init_guess,np.ndarray): 
         # init guess -- increase bond dimension using svd init guess
         leg_ind = PEPX.leg2ind(connect_list[0])
         if sub_peps_gl[0,0].shape[leg_ind] < DMAX:
             gam_opt_qr,lam_opt_qr,errs = PEPX_GL.compress_GL_list(gam_dt_qr,lam_dt_qr,DMAX=DMAX,normalize=normalize)
         else:
             peps_opt_gl = init_guess
             gam_list, lam_list, axT_invs = PEPX_GL.get_sites(init_guess,ind0,connect_list)
             gam_opt_qr, lam_opt_qr, q02, qL2 = PEPX_GL.full_to_qr_GL_list(gam_list,lam_list,num_io=num_io)
                 # just throw out these q's. not great but it's just an initial guess anyways...
             if flatten:
                 peps_opt_qr = [tf.reshape(m,'...,ii,i') for m in gam_opt_qr]

    elif init_guess in ['rand','random']:
         raise(NotImplementedError), 'random init state for alsq not implemented'

    else:     # init_guess is None or init_guess in ['su','SU','svd']:
         gam_opt_qr,lam_opt_qr,errs = PEPX_GL.compress_GL_list(gam_dt_qr,lam_dt_qr,DMAX=DMAX,normalize=normalize)
             # already flattened


    # print 'checking compress GL vs simple update full'
    # gam_opt_T, lam_opt_T, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=DMAX,num_io=num_io,direction=0,
    #                                            normalize=normalize)
    # if flatten:  gam_opt_T = [tf.reshape(m,'...,ii,i') for m in gam_opt_T]
 
    # gam_opt_qr_c, lam_opt_qr_c = PEPX_GL.canonicalize_GL_list(gam_opt_qr,lam_opt_qr,num_io=num_io)
    # gam_opt_Tc, lam_opt_Tc = PEPX_GL.canonicalize_GL_list(gam_opt_T,lam_opt_T,num_io=num_io)

    # gam_opt_c, lam_opt_c = PEPX_GL.qr_to_full_GL_list(gam_opt_qr_c,lam_opt_qr_c,q0,qL,lam0,lamL)

    # for i in range(len(gam_opt_c)):
    #     print 'gam', i, np.linalg.norm(gam_opt_c[i]-gam_opt_Tc[i])
    # for i in range(len(lam_opt_c)):
    #     print 'lam', i, np.linalg.norm(lam_opt_c[i]-lam_opt_Tc[i])

    # tens1 = tf.dMult('DM',lam_opt_c[0], tf.dMult('MD',gam_opt_c[0],lam_opt_c[1]))
    # tens1 = np.tensordot( tens1, tf.dMult('MD',gam_opt_c[1],lam_opt_c[2]), axes=(-1,0))
    # tens2 = tf.dMult('DM',lam_opt_Tc[0], tf.dMult('MD',gam_opt_Tc[0],lam_opt_Tc[1]))
    # tens2 = np.tensordot( tens2, tf.dMult('MD',gam_opt_Tc[1],lam_opt_Tc[2]) , axes=(-1,0))

    # print 'tens', np.linalg.norm(tens1-tens2)

    ########

    site_list = [(xs[i],ys[i]) for i in range(len(xs))]
    temp_xs = [m[0] for m in site_list]
    temp_ys = [m[1] for m in site_list]
    site_conn = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)

    gam_opt_qr, lam_opt_qr = alsq_redenv_update((L1,L2),site_list,site_conn,
                                                gam_dt_qr,lam_dt_qr, gam_opt_qr,lam_opt_qr,
                                                norm_env,norm_env, normalize=normalize,
                                                max_it=max_it,conv_tol=conv_tol)
    if flatten:   # unflatten
        dbs = sub_peps_gl.phys_bonds
        # print [m.shape for m in gam_opt_qr]
        temp = []
        for i in range(len(site_list)):
            m    = gam_opt_qr[i]
            idx  = site_list[i]
            temp += [np.reshape(m, m.shape[:-len(dbs[idx])]+dbs[idx]+m.shape[-1:])]
        gam_opt_qr = temp
        # print [m.shape for m in gam_opt_qr]

    gam_opt, lam_opt = PEPX_GL.qr_to_full_GL_list(gam_opt_qr,lam_opt_qr,q0,qL,lam0,lamL)
    peps_opt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_opt,lam_opt,axT_invs) 

    # if flatten:
    #     peps_opt_gl = PEPX_GL.unflatten(peps_opt_gl,dbs)

    return peps_opt_gl


def alsq_redenv_update(Ls,idx_list,site_conn, gam_dt_qr,lam_dt_qr, gam_opt_qr,lam_opt_qr, norm_env_m,norm_env_g, 
                       normalize=False,ensure_pos=False,max_it=500,conv_tol=1.0e-8):
    ''' optimize each tens in peps_list, given environment
    '''

    assert (Ls == (1,2) or Ls == (2,1)), 'qr + env only can be done for 2x1 or 1x2 sub_peps'
    # num_io = sub_peps_gl[0,0].ndim-4
  
    ## dict of gam list with keys as positions in lattice.  tensors transposed to be qdr
    ## (or liord if not 2site block )not implemented))
    
    # print 'dt', [m.shape for m in gam_dt_qr], [m.shape for m in lam_dt_qr]
    # print 'opt',[m.shape for m in gam_opt_qr], [m.shape for m in lam_opt_qr]
    
    gl_dict_dt  = qr_list_to_dict(gam_dt_qr ,lam_dt_qr ,idx_list)
    gl_dict_opt = qr_list_to_dict(gam_opt_qr,lam_opt_qr,idx_list)
    old_gl_dict_opt = gl_dict_opt

    # print 'alsq redenv', lam_opt_qr


    ## start alsq optimization
    not_converged = True
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0

    site_list = idx_list
    temp_xs = [m[0] for m in site_list]
    temp_ys = [m[1] for m in site_list]
    site_conn = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)
    
    # while (not_converged or it < 3) and it < max_it:
    while (not_converged) and it < max_it:     # 200726 change

        for xj in range(len(site_list)):

            idx = site_list[xj]
            iso_leg = site_conn[xj]

            # if build_env:
            env_xx = ENV_GL.red_embed_sites_xx_env( Ls, gl_dict_opt, lam_opt_qr[1],
                                                    gl_dict_opt, lam_opt_qr[1], norm_env_m, idx ) 
            env_x  = ENV_GL.red_embed_sites_xo_env( Ls, gl_dict_opt, lam_opt_qr[1],
                                                    gl_dict_dt,  lam_dt_qr[1],  norm_env_g, idx )
                            # need to take complex conj in function

            # temp = np.einsum('qQxX,qdx,QdX',env_xx,np.conj(tf.dMult('MD',gl_dict_opt[idx],lam_opt_qr[1])),
            #                                 tf.dMult('MD',gl_dict_opt[idx],lam_opt_qr[1]))
            # print 'ovlp', temp, np.sqrt(temp)

            r_opt = red_optimize_site(env_xx,env_x)   # ,ensure_pos=ensure_pos)   # includes lams (d at end) 
            # r_opt = optimize_site_constrained(env_xx,env_x,do_reduced=qr_reduce)
            r_opt = np.moveaxis( r_opt, -1, -2 )   # qrd -> qdr

            # print 'alsq redenv', lam_opt_qr

            ## only change gamma (keep lambda from dt svd) 
            # peps_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,peps_opt_gl.lambdas[idx])

            ## do svd on site_opt + next site to keep it close to GL form
            site_opt_gam = tf.dMult('MD',r_opt,1./lam_opt_qr[1])   #qdr/lam_r
            if xj == 0:                          gam_opt_qr[xj] = site_opt_gam
            elif xj == len(site_list)-1:         gam_opt_qr[xj] = np.swapaxes(site_opt_gam, 0, -1)  #qdr->ldq
            else:  raise(ValueError),'xj should be only 0 or 1 for 2-site block'

            gam_opt_qr, lam_opt_qr = PEPX_GL.canonicalize_GL_list(gam_opt_qr,lam_opt_qr,normalize=False)
            gl_dict_opt = qr_list_to_dict(gam_opt_qr,lam_opt_qr,idx_list)

            # print 'canon gam', lam_opt_qr

            if np.any( [np.linalg.norm(m) > 10 for m in lam_opt_qr[1:-1]] ):
                print 'large lambda', idx, iso_leg, [np.linalg.norm(m) for m in lam_opt_qr]
                print 'large lambda', idx, iso_leg, [np.linalg.norm(m) for m in lam_dt_qr]
                print [np.linalg.norm(m) for m in gam_opt_qr]
                raise RuntimeWarning('large lambda, '+str(idx))

        ############

        regularize = True
        if regularize:    ## blows up if no regularization is done
            gam_opt_qr, lam_opt_qr = PEPX_GL.regularize_GL_list(gam_opt_qr,lam_opt_qr)
            # peps_opt_gl = PEPX_GL.regularize(peps_opt_gl)
            # print 'regularize', [np.linalg.norm(m) for m in gam_opt], lam_opt

        ## build_env:
        opt_norm = ENV_GL.red_embed_sites_norm_env(Ls, gl_dict_opt, lam_opt_qr[1], norm_env_m)
        fidelity = ENV_GL.red_embed_sites_ovlp_env(Ls, gl_dict_opt, lam_opt_qr[1], gl_dict_dt,
                                                   lam_dt_qr[1], norm_env_g) / opt_norm
        # print 'opt', opt_norm, fidelity
        # exit()


        if np.isnan(opt_norm):
            print 'redenv update: nan opt norm!', opt_norm
            print 'ovlp', ENV_GL.red_embed_sites_ovlp_env(Ls, gl_dict_opt, lam_opt_qr[1], 
                                                      gl_dict_opt, lam_opt_qr[1], norm_env_m)

            sqdim = int(np.sqrt(np.prod(norm_env_m.shape)))
            axT = np.arange(norm_env_m.ndim).reshape(2,-1).T.reshape(-1)
            env_m = np.transpose(norm_env_m,axT).reshape(sqdim,sqdim)

            evals,evecs = np.linalg.eig(env_m)
            print evals

            # print 'dt norm'
            # print ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
           
            # print 'init norm'
            # if flatten:    sub_peps_gl = PEPX_GL.flatten(sub_peps_gl)
            # print ENV_GL.embed_sites_norm(sub_peps_gl, env_list)

            raise RuntimeError('optimal norm is nan')


        # print 'fidelity', it, fidelity, opt_norm

        if np.imag(fidelity) > np.real(fidelity):
            print fidelity, opt_norm
            print 'imag opt', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
            print 'real opt', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
            print 'imag dt',  [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
            print 'real dt',  [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
            if build_env:  print np.linalg.norm(norm_env)
            raise RuntimeWarning('imag fidelity or norm')


        not_converged = np.abs(fidelity - prev_fidelity) > conv_tol #1.0e-6  #1.0e-8   #1.0e-10
        fid_diff = fidelity-prev_fidelity

        if (prev_fidelity > fidelity+1.0e-12):
            gl_dict_opt = old_gl_dict_opt
            print 'fidelity went down or alsq diff went up, reset opt gl'
            break

        # prev_alsq_diff = alsq_diff
        prev_fidelity   = fidelity
        old_gl_dict_opt = gl_dict_opt

        it += 1

    if not_converged:  print 'not converged', it, fid_diff

    if normalize:
        gam_opt_qr, lam_opt_qr = PEPX_GL.mul_GL_list(gam_opt_qr,lam_opt_qr,1./opt_norm)
        # peps_opt_gl = PEPX_GL.mul(1./opt_norm, peps_opt_gl)

    return gam_opt_qr,lam_opt_qr
    

#####################################################
####  reduced update for 2x2 (isolate 2 sites)  #####
#####################################################

def alsq_redenv_2x2(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                    init_guess=None, ensure_pos=False, normalize=False, flatten=False, max_it=500, sub_max_it=50,
                    conv_tol=1.e-8, sub_conv_tol=1.e-8):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        xy_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over

        reduce the 2x2 problem to a set of 1x2 problems; insert sites with compression to X.
    '''

    L1,L2 = sub_peps_gl.shape
    num_io = sub_peps_gl[0,0].ndim-4
    xs,ys = xy_list

    ind0 = (xy_list[0][0],xy_list[1][0])
    gam_list, lam_list, axT_invs = PEPX_GL.get_sites(sub_peps_gl,ind0,connect_list)

    # apply trotter and then svd
    gam_dt, lam_dt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=-1,direction=0,num_io=num_io,
                                         normalize=normalize)
    peps_dt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_dt,lam_dt,axT_invs) 
    # print 'peps dt', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)], [m.shape for m in trotter_block]

    if flatten:
        dbs = peps_dt_gl.phys_bonds
        peps_dt_gl = PEPX_GL.flatten(peps_dt_gl)
        # peps_opt_gl = PEPX_GL.flatten(peps_opt_gl)
        # print 'flattened', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)]

    if normalize:
        ## if build_env:        dt_norm = ENV_GL.embed_sites_norm_env(peps_dt_gl, norm_env)   # env is without lambdas
        ## else:                dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
        # print 'dt norm', dt_norm
        dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
        peps_dt_gl = PEPX_GL.mul(1./dt_norm, peps_dt_gl)


    if isinstance(init_guess,np.ndarray): 
        # init guess -- increase bond dimension using svd init guess
        leg_ind = PEPX.leg2ind(connect_list[0])
        if sub_peps_gl[0,0].shape[leg_ind] < DMAX:
            # peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)
            gam_opt, lam_opt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=DMAX,direction=0,
                                                   num_io=num_io,normalize=normalize)
            peps_opt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_opt,lam_opt,axT_invs) 
        else:
            peps_opt_gl = init_guess

        if flatten:
            peps_opt_gl  = PEPX_GL.flatten(peps_opt_gl)

    elif init_guess in ['rand','random']:
        leg_ind_o = [PEPX_GL.leg2ind(c_leg) for c_leg in connect_list]+[None]
        leg_ind_i = [None]+[PEPX_GL.leg2ind(PEPX_GL.opposite_leg(c_leg)) for c_leg in connect_list]
        shape_array = np.empty((L1,L2),dtype=tuple)
        for i in range(len(peps_list)):
            ind = (xs[i],ys[i])
            rand_shape = list(peps_list[i].shape)
            if leg_ind_o[i] is not None:     rand_shape[leg_ind_o[i]] = DMAX
            if leg_ind_i[i] is not None:     rand_shape[leg_ind_i[i]] = DMAX
            shape_array[ind] = tuple(rand_shape)

        raise(NotImplementedError), 'random init state for alsq not implemented'

    else:     # init_guess is None or init_guess in ['su','SU','svd']:
        # peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)
        gam_opt, lam_opt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=DMAX,num_io=num_io,
                                               direction=0,normalize=normalize)
        peps_opt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_opt,lam_opt,axT_invs) 

        if flatten:
            peps_opt_gl  = PEPX_GL.flatten(peps_opt_gl)



    not_converged = True
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0

    bL,bI,bO,bR = env_list
    new_bounds = ['R','O','L','I']
    old_peps_opt_gl = peps_opt_gl

    XMAX = -1

    while (not_converged or it < 3) and it < max_it:

        for b_dir in new_bounds:

          if b_dir == 'R':

            bL_ = bL[:]
            bI_ = bI[:1]
            bO_ = bO[:1]
            bR_m,err = ENV_GL.get_next_subboundary_R(bR,bI[1],np.conj(peps_opt_gl[:,1]),peps_opt_gl[:,1],bO[1],XMAX,0)
            bR_g,err = ENV_GL.get_next_subboundary_R(bR,bI[1],np.conj(peps_opt_gl[:,1]),peps_dt_gl [:,1],bO[1],XMAX,0)

            env_list_g = [bL_,bI_,bO_,bR_g]
            env_list_m = [bL_,bI_,bO_,bR_m]

            sub_opt = peps_opt_gl[:,:1]
            sub_dt  = peps_dt_gl [:,:1]

            site_list = [(0,0),(1,0)]
            site_conn = 'oi'
            sub_ind0 = (0,0)

          elif b_dir == 'L':

            bL_m,err = ENV_GL.get_next_subboundary_L(bL,bI[0],np.conj(peps_opt_gl[:,0]),peps_opt_gl[:,0],bO[0],XMAX,0)
            bL_g,err = ENV_GL.get_next_subboundary_L(bL,bI[0],np.conj(peps_opt_gl[:,0]),peps_dt_gl [:,0],bO[0],XMAX,0)
            bI_ = bI[1:]
            bO_ = bO[1:]
            bR_ = bR[:]

            env_list_g = [bL_g,bI_,bO_,bR_]
            env_list_m = [bL_m,bI_,bO_,bR_]

            sub_opt = peps_opt_gl[:,1:]
            sub_dt  = peps_dt_gl [:,1:]

            site_list = [(0,0),(1,0)]
            site_conn = 'oi'
            sub_ind0 = (0,1)

          elif b_dir == 'O':

            bL_ = bL[:1]
            bI_ = bI[:]
            bO_m,err = ENV_GL.get_next_subboundary_O(bO,bL[1],np.conj(peps_opt_gl[1,:]),peps_opt_gl[1,:],bR[1],XMAX,0)
            bO_g,err = ENV_GL.get_next_subboundary_O(bO,bL[1],np.conj(peps_opt_gl[1,:]),peps_dt_gl [1,:],bR[1],XMAX,0)
            bR_ = bR[:1]

            env_list_g = [bL_,bI_,bO_g,bR_]
            env_list_m = [bL_,bI_,bO_m,bR_]

            sub_opt = peps_opt_gl[:1,:]
            sub_dt  = peps_dt_gl [:1,:]

            site_list = [(0,0),(0,1)]
            site_conn = 'rl'
            sub_ind0 = (0,0)

          elif b_dir == 'I':

            bL_ = bL[1:]
            bI_m,err = ENV_GL.get_next_subboundary_I(bI,bL[0],np.conj(peps_opt_gl[0,:]),peps_opt_gl[0,:],bR[0],XMAX,0)
            bI_g,err = ENV_GL.get_next_subboundary_I(bI,bL[0],np.conj(peps_opt_gl[0,:]),peps_dt_gl [0,:],bR[0],XMAX,0)
            bO_ = bO[:]
            bR_ = bR[1:]

            env_list_g = [bL_,bI_g,bO_,bR_]
            env_list_m = [bL_,bI_m,bO_,bR_]

            sub_opt = peps_opt_gl[1:,:]
            sub_dt  = peps_dt_gl [1:,:]

            site_list = [(0,0),(0,1)]
            site_conn = 'rl'
            sub_ind0 = (1,0)

          else:
              raise(IndexError), 'choose valid direction of square to contract into'


          L1_,L2_ = sub_opt.shape

          # print type(sub_dt), [(idx,m.shape) for idx,m in np.ndenumerate(sub_dt)],sub_dt.phys_bonds,sub_connect_list

          gam_dt, lam_dt, axT_invs = PEPX_GL.get_sites(sub_dt,(0,0),site_conn[0])
          gam_dt_qr, lam_dt_qr, q0t, qLt = PEPX_GL.full_to_qr_GL_list(gam_dt,lam_dt)
          lam0t, lamLt = lam_dt[0],lam_dt[-1]

          # print 'gam dt', [m.shape for m in gam_dt], site_conn

          gam_opt, lam_opt, axT_invs = PEPX_GL.get_sites(sub_opt,(0,0),site_conn[0])
          gam_opt_qr, lam_opt_qr, q0o, qLo = PEPX_GL.full_to_qr_GL_list(gam_opt,lam_opt)
          lam0o, lamLo = lam_opt[0],lam_opt[-1]

          ## if build_env:

          # print [[m.shape for m in b] for b in env_list_g]
          # print [[m.shape for m in b] for b in env_list_m]

          # print 'opt',[(idx,m.shape) for idx,m in np.ndenumerate(sub_opt)]
          # print gam_opt_qr[0].shape, gam_opt_qr[-1].shape
          # print q0o.shape, qLo.shape, [m.shape for m in gam_opt_qr]

          # print 'dt', [(idx,m.shape) for idx,m in np.ndenumerate(sub_dt)]
          # print gam_dt_qr[0].shape, gam_dt_qr[-1].shape
          # print q0t.shape, qLt.shape
          # print [m.shape for m in gam_dt_qr]

          # ensure_pos = False
          # print 'enusre pos', ensure_pos
          norm_env_g = ENV_GL.build_env_qr(env_list_g,[q0o,qLo],[q0t,qLt],ensure_pos=False)   # no lambdas
          norm_env_m = ENV_GL.build_env_qr(env_list_m,[q0o,qLo],[q0o,qLo],ensure_pos=ensure_pos)   # no lambdas

          ## obtain optimal overlap (up to sub_max_it iterations)
          # print 'call update', b_dir, sub_ind0, site_conn
          # print 'dt', [m.shape for m in gam_dt_qr]
          # print 'opt', [m.shape for m in gam_opt_qr]
          gam_opt_qr, lam_opt_qr = alsq_redenv_update((L1_,L2_),site_list,site_conn,
                                                      gam_dt_qr,lam_dt_qr, gam_opt_qr,lam_opt_qr,
                                                      norm_env_m,norm_env_g,
                                                      normalize=normalize,max_it=sub_max_it,conv_tol=sub_conv_tol)
          # print 'opt2', [m.shape for m in gam_opt_qr]

          gam_opt, lam_opt = PEPX_GL.qr_to_full_GL_list(gam_opt_qr,lam_opt_qr,q0o,qLo,lam0o,lamLo)
          # peps_opt_gl = PEPX_GL.set_sites(peps_opt_gl,sub_ind0,site_conn[0],gam_opt,lam_opt,axT_invs) 
          sub_opt = PEPX_GL.set_sites(sub_opt,(0,0),site_conn[0],gam_opt,lam_opt,axT_invs) 

          # print [(idx,m.shape) for idx,m in np.ndenumerate(peps_opt_gl)]

          # exit()

          # ######## check red env norm/ovlp vs build env ###########
          # temp_opt = [np.sqrt(1)*m for m in gam_opt_qr]
          # gam_dict_opt = qr_list_to_dict(temp_opt,lam_opt_qr,site_list)
          # gam_dict_dt  = qr_list_to_dict(gam_dt_qr ,lam_dt_qr ,site_list)
          # opt_norm = ENV_GL.red_embed_sites_norm_env((L1_,L2_), gam_dict_opt, lam_opt[1], norm_env_m)
          # fidelity = ENV_GL.red_embed_sites_ovlp_env((L1_,L2_), gam_dict_opt, lam_opt[1],
          #                                                       gam_dict_dt , lam_dt[1] , norm_env_g)/opt_norm

          # temp_gl = PEPX_GL.mul(1,sub_opt.copy())
          # opt_norm_1 = ENV_GL.embed_sites_norm(temp_gl, env_list_m)
          # fidelity_1 = ENV_GL.embed_sites_ovlp(np.conj(temp_gl),sub_dt,env_list_g)/opt_norm_1

          # # opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list)
	  # # fidelity = ENV_GL.embed_sites_ovlp(np.conj(peps_opt_gl),peps_dt_gl,env_list)/opt_norm

          # if np.abs(fidelity_1 - fidelity) > 1.0e-10:
          #     print 'sub fidelity different', (fidelity - fidelity_1)/fidelity
          #     print 'sub opt norm different', (opt_norm - opt_norm_1)/opt_norm
          #     print [np.linalg.norm(m) for m in gam_opt_qr], [np.linalg.norm(m) for m in lam_opt_qr]
          #     print lam_opt_qr, lam_dt_qr
          #     print b_dir, site_list


        # contracted b_dir to env; update other row/col
        if   b_dir == 'R':          peps_opt_gl[:,:1] = sub_opt
        elif b_dir == 'L':          peps_opt_gl[:,1:] = sub_opt
        elif b_dir == 'O':          peps_opt_gl[:1,:] = sub_opt
        elif b_dir == 'I':          peps_opt_gl[1:,:] = sub_opt
        else:   raise(ValueError),'b_dir not a valid direction'


        # these two methods are different if XMAX != -1 above (in getting sub_pepx)
        if XMAX <= 0:
          gam_dict_opt = qr_list_to_dict(gam_opt_qr,lam_opt_qr,site_list)
          gam_dict_dt  = qr_list_to_dict(gam_dt_qr ,lam_dt_qr ,site_list)
          opt_norm = ENV_GL.red_embed_sites_norm_env((L1_,L2_), gam_dict_opt, lam_opt[1], norm_env_m)
          fidelity = ENV_GL.red_embed_sites_ovlp_env((L1_,L2_), gam_dict_opt, lam_opt[1],
                                                                gam_dict_dt , lam_dt[1] , norm_env_g)/opt_norm

          # opt_norm_1 = ENV_GL.embed_sites_norm(peps_opt_gl, env_list)
          # fidelity_1 = ENV_GL.embed_sites_ovlp(np.conj(peps_opt_gl),peps_dt_gl,env_list)/opt_norm_1

          # if np.abs(fidelity_1 - fidelity) > 1.0e-10:
          #     print 'fidelity different', fidelity - fidelity_1
          #     print 'opt norm different', opt_norm - opt_norm_1
          #     print [np.linalg.norm(m) for m in gam_opt_qr], [np.linalg.norm(m) for m in lam_opt_qr]
          #     print lam_opt_qr, lam_dt_qr
          #     print b_dir, site_list

          # if np.abs(opt_norm_1 - opt_norm) > 1.0e-10:
          #     print 'opt norm different', opt_norm - opt_norm_1
          #     exit()

          # print '2x2 fidelity, opt'
          # print 'opt', opt_norm, opt_norm_1
          # print 'fid', fidelity, fidelity_1
          # # exit()

        else:
          opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list)
	  fidelity = ENV_GL.embed_sites_ovlp(np.conj(peps_opt_gl),peps_dt_gl,env_list)/opt_norm


        regularize = True
        if regularize:    ## blows up if no regularization is done
            peps_opt_gl = PEPX_GL.regularize(peps_opt_gl)
            oldc_x, oldc_xx = None,None


        if np.isnan(opt_norm):
            print 'alsq redenv 2x2: nan opt norm!', opt_norm

            print 'dt norm'
            print ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
            
            print 'init norm'
            if flatten:    sub_peps_gl = PEPX_GL.flatten(sub_peps_gl)
            print ENV_GL.embed_sites_norm(sub_peps_gl, env_list)

            raise RuntimeError('optimal norm is nan')


        if np.imag(fidelity) > np.real(fidelity):
            print fidelity, opt_norm
            print 'imag opt', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
            print 'real opt', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
            print 'imag dt',  [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
            print 'real dt',  [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
            print np.linalg.norm(norm_env_g), np.linalg.norm(norm_env_m)
            raise RuntimeWarning('imag fidelity or norm')


        not_converged = np.abs(fidelity - prev_fidelity) > conv_tol #1.0e-6  #1.0e-8  #1.0e-10
        fid_diff = fidelity-prev_fidelity
        # print 'fid diff', fid_diff
        # print 'fidelity', it, fidelity, opt_norm, fid_diff

        if (prev_fidelity > fidelity+1.0e-12):
            peps_opt_gl = old_peps_opt_gl
            break

        # prev_alsq_diff = alsq_diff
        prev_fidelity   = fidelity
        old_peps_opt_gl = peps_opt_gl

        it += 1

    if not_converged:  print 'not converged', it, fid_diff

    if normalize:
        peps_opt_gl = PEPX_GL.mul(1./opt_norm, peps_opt_gl)

    if flatten:
        peps_opt_gl = PEPX_GL.unflatten(peps_opt_gl,dbs)


    return peps_opt_gl


#####################################################
#### alternate update for 2x2 (isolate 2 sites) #####
#####################################################

def alsq_block_update_2x2(sub_peps_gl,connect_list,xy_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                          init_guess=None, ensure_pos=False, normalize=False, regularize=True, build_env=False,
                          qr_reduce=False, flatten=False, max_it=500, sub_max_it=50, conv_tol=1.e-8, sub_conv_tol=1.e-8):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        xy_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over

        reduce the 2x2 problem to a set of 1x2 problems; insert sites with compression to X.
    '''

    L1,L2 = sub_peps_gl.shape
    num_io = sub_peps_gl[0,0].ndim-4
    xs,ys = xy_list

    ind0 = (xy_list[0][0],xy_list[1][0])
    gam_list, lam_list, axT_invs = PEPX_GL.get_sites(sub_peps_gl,ind0,connect_list)

    # apply trotter and then svd
    gam_dt, lam_dt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=-1,num_io=num_io,direction=0,
                                         normalize=normalize)
    peps_dt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_dt,lam_dt,axT_invs) 
    # print 'peps dt', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)], [m.shape for m in trotter_block]

    if flatten:
        dbs = peps_dt_gl.phys_bonds
        peps_dt_gl = PEPX_GL.flatten(peps_dt_gl)
        # peps_opt_gl = PEPX_GL.flatten(peps_opt_gl)
        # print 'flattened', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)]

    if normalize:
        ## if build_env:        dt_norm = ENV_GL.embed_sites_norm_env(peps_dt_gl, norm_env)   # env is without lambdas
        ## else:                dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
        # print 'dt norm', dt_norm
        dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
        peps_dt_gl = PEPX_GL.mul(1./dt_norm, peps_dt_gl)


    if isinstance(init_guess,np.ndarray): 
         # init guess -- increase bond dimension using svd init guess
         leg_ind = PEPX.leg2ind(connect_list[0])
         if sub_peps_gl[0,0].shape[leg_ind] < DMAX:
             # peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)
             gam_opt, lam_opt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=DMAX,num_io=num_io,
                                                    direction=0,normalize=normalize)
             peps_opt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_opt,lam_opt,axT_invs) 
         else:
             peps_opt_gl = init_guess

         if flatten:
             peps_opt_gl  = PEPX_GL.flatten(peps_opt_gl)

    elif init_guess in ['rand','random']:
         leg_ind_o = [PEPX_GL.leg2ind(c_leg) for c_leg in connect_list]+[None]
         leg_ind_i = [None]+[PEPX_GL.leg2ind(PEPX_GL.opposite_leg(c_leg)) for c_leg in connect_list]
         shape_array = np.empty((L1,L2),dtype=tuple)
         for i in range(len(peps_list)):
             ind = (xs[i],ys[i])
             rand_shape = list(peps_list[i].shape)
             if leg_ind_o[i] is not None:     rand_shape[leg_ind_o[i]] = DMAX
             if leg_ind_i[i] is not None:     rand_shape[leg_ind_i[i]] = DMAX
             shape_array[ind] = tuple(rand_shape)

         raise(NotImplementedError), 'random init state for alsq not implemented'

    else:     # init_guess is None or init_guess in ['su','SU','svd']:
         # peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)
         gam_opt, lam_opt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=DMAX,direction=0,num_io=num_io,
                                                normalize=normalize)
         peps_opt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_opt,lam_opt,axT_invs) 

         if flatten:
             peps_opt_gl  = PEPX_GL.flatten(peps_opt_gl)



    not_converged = True
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0

    bL,bI,bO,bR = env_list
    new_bounds = ['R','O','L','I']
    old_peps_opt_gl = peps_opt_gl
    
    # while (not_converged or it < 3) and it < max_it:
    while (not_converged) and it < max_it:     # 200726 change

        for b_dir in new_bounds:

            if b_dir == 'R':

              bL_ = bL[:]
              bI_ = bI[:1]
              bO_ = bO[:1]
              bR_m,err = ENV_GL.get_next_subboundary_R(bR,bI[1],np.conj(peps_opt_gl[:,1]),peps_opt_gl[:,1],bO[1],XMAX)
              bR_g,err = ENV_GL.get_next_subboundary_R(bR,bI[1],np.conj(peps_opt_gl[:,1]),peps_dt_gl [:,1],bO[1],XMAX)

              env_list_g = [bL_,bI_,bO_,bR_g]
              env_list_m = [bL_,bI_,bO_,bR_m]

              sub_opt = peps_opt_gl[:,:1]
              sub_dt  = peps_dt_gl [:,:1]

              site_list = [(0,0),(1,0)]
              site_conn = 'oi'

            elif b_dir == 'L':

              bL_m,err = ENV_GL.get_next_subboundary_L(bL,bI[0],np.conj(peps_opt_gl[:,0]),peps_opt_gl[:,0],bO[0],XMAX)
              bL_g,err = ENV_GL.get_next_subboundary_L(bL,bI[0],np.conj(peps_opt_gl[:,0]),peps_dt_gl [:,0],bO[0],XMAX)
              bI_ = bI[1:]
              bO_ = bO[1:]
              bR_ = bR[:]

              env_list_g = [bL_g,bI_,bO_,bR_]
              env_list_m = [bL_m,bI_,bO_,bR_]

              sub_opt = peps_opt_gl[:,1:]
              sub_dt  = peps_dt_gl [:,1:]

              site_list = [(0,0),(1,0)]
              site_conn = 'oi'

            elif b_dir == 'O':

              bL_ = bL[:1]
              bI_ = bI[:]
              bO_m,err = ENV_GL.get_next_subboundary_O(bO,bL[1],np.conj(peps_opt_gl[1,:]),peps_opt_gl[1,:],bR[1],XMAX)
              bO_g,err = ENV_GL.get_next_subboundary_O(bO,bL[1],np.conj(peps_opt_gl[1,:]),peps_dt_gl [1,:],bR[1],XMAX)
              bR_ = bR[:1]

              env_list_g = [bL_,bI_,bO_g,bR_]
              env_list_m = [bL_,bI_,bO_m,bR_]

              sub_opt = peps_opt_gl[:1,:]
              sub_dt  = peps_dt_gl [:1,:]

              site_list = [(0,0),(0,1)]
              site_conn = 'rl'

            elif b_dir == 'I':

              bL_ = bL[1:]
              bI_m,err = ENV_GL.get_next_subboundary_I(bI,bL[0],np.conj(peps_opt_gl[0,:]),peps_opt_gl[0,:],bR[0],XMAX)
              bI_g,err = ENV_GL.get_next_subboundary_I(bI,bL[0],np.conj(peps_opt_gl[0,:]),peps_dt_gl [0,:],bR[0],XMAX)
              bO_ = bO[:]
              bR_ = bR[1:]

              env_list_g = [bL_,bI_g,bO_,bR_]
              env_list_m = [bL_,bI_m,bO_,bR_]

              sub_opt = peps_opt_gl[1:,:]
              sub_dt  = peps_dt_gl [1:,:]

              site_list = [(0,0),(0,1)]
              site_conn = 'rl'

            else:
                raise(IndexError), 'choose valid direction of square to contract into'


            L1_,L2_ = sub_opt.shape

            if build_env:
                norm_env_g = ENV_GL.build_env(env_list_g,ensure_pos=ensure_pos)   # no lambdas
                norm_env_m = ENV_GL.build_env(env_list_m,ensure_pos=ensure_pos)   # no lambdas


            sub_opt = alsq_gen_update((L1_,L2_), [site_list], [site_conn], sub_dt, sub_opt, env_list_m, env_list_g,
                                      normalize=normalize, regularize=regularize, ensure_pos=ensure_pos,
                                      build_env=build_env,qr_reduce=qr_reduce, max_it=sub_max_it, conv_tol=sub_conv_tol)

            # print 'init'
            # print 'opt gl', [(idx,m.shape) for idx,m in np.ndenumerate(peps_opt_gl)]
            # print 'init envs', [[x.shape for x in m] for m in env_list]
            # print 'b dir', b_dir, it
            # print 'opt', [(idx,m.shape) for idx,m in np.ndenumerate(sub_opt)]
            # print 'dt ', [(idx,m.shape) for idx,m in np.ndenumerate(sub_dt)]
            # print 'envs m', [[x.shape for x in m] for m in env_list_m]
            # print 'envs g', [[x.shape for x in m] for m in env_list_g]

            # sub_not_converged = True
            # sub_old_opt = sub_opt
            # sub_fidelities = []
            # sub_prev_alsq_diff = 100
            # sub_prev_fidelity  = 0
            # sub_it = 0

            # while (sub_not_converged or sub_it < 3) and sub_it < 50:

            #     for xj in range(len(site_list)):

            #         idx = site_list[xj]
            #         iso_leg = site_conn[xj]

            #         if qr_reduce and (xj == 0 or xj == len(site_list)-1):

            #             env_xx,qu_idx,qd_idx,axT_inv = \
            #                    ENV_GL.red_embed_sites_xx(np.conj(sub_opt),sub_opt,env_list_m,idx,iso_leg,
            #                                              return_corners=False)
            #             env_x = \
            #                    ENV_GL.red_embed_sites_xo(np.conj(sub_opt),sub_dt,env_list_g,idx,iso_leg,qu_idx,qd_idx,
            #                                              return_corners=False)

            #             r_opt = red_optimize_site(env_xx,env_x,ensure_pos=ensure_pos)   # includes lams
            #             # r_opt = optimize_site_constrained(env_xx,env_x,do_reduced=qr_reduce)

            #             site_opt = PEPX.QR_contract(qd_idx,r_opt,axT_inv,d_end=True)

            #         else:
            #             # print 'opt',[ma.shape for xx, ma in np.ndenumerate(peps_opt_gl)]
            #             # print 'dt', [ma.shape for xx, ma in np.ndenumerate(peps_dt_gl)]
            #             env_x  = ENV_GL.embed_sites_xo(np.conj(sub_opt),sub_dt ,env_list_g,idx,return_corners=False)
            #             env_xx = ENV_GL.embed_sites_xx(np.conj(sub_opt),sub_opt,env_list_m,idx,return_corners=False)

            #             site_opt = optimize_site(env_xx,env_x,ensure_pos=ensure_pos)   # includes lams
            #             # site_opt = optimize_site_2(env_xx,env_x,do_lsq=True)   # includes lams
            #             # site_opt = optimize_site_constrained(env_xx,env_x)   # includes lams
            #             # site_opt = optimize_site_constrained_2(env_xx,env_x)   # includes lams
            #             # site_opt = optimize_site_constrained_3(env_xx,env_x)   # includes lams

            #         # print np.linalg.norm(site_opt), np.linalg.norm( PEPX_GL.get_site(peps_dt_gl,idx) )


            #         ## only change gamma (keep lambda from dt svd)
            #         # peps_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,peps_opt_gl.lambdas[idx])

            #         ## do svd on site_opt + next site to keep it close to GL form
            #         sub_opt[idx] = PEPX_GL.remove_lam_from_site(site_opt,sub_opt.lambdas[idx])
            #         gs, ls, axTs = PEPX_GL.get_sites(sub_opt,idx,iso_leg)
            #         gs_, ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,normalize=False)
            #         sub_opt = PEPX_GL.set_sites(sub_opt,idx,iso_leg,gs_,ls_,axTs)

            #         # if normalize:
            #         #     opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list, old_corners=oldc_xx)
            #         #     gs_,ls_ = PEPX_GL.mul_GL_list(gs_,ls_,1./opt_norm)
            #         #     peps_opt_gl = PEPX_GL.set_sites(peps_opt_gl,idx,iso_leg,gs_,ls_,axTs)
            #         

            #         if np.any( [np.linalg.norm(sub_opt.lambdas[xx]) > 10 for xx in np.ndindex(L1_,L2_,4)] ):
            #             print 'large lambda', idx, iso_leg, [np.linalg.norm(sub_opt.lambdas[xx]) for xx in
            #                                                      np.ndindex(L1_,L2_,4)]
            #             print 'large lambda', idx, iso_leg, [np.linalg.norm(sub_dt.lambdas[xx]) for xx in
            #                                                      np.ndindex(L1_,L2_,4)]
            #             print [np.linalg.norm(m) for idx, m in np.ndenumerate(sub_opt)]
            #             opt_norm = ENV_GL.embed_sites_norm(sub_opt, env_list_g, old_corners=None)
            #             print opt_norm
            #             raise RuntimeWarning('large lambda, '+str(idx))

            #     ############

            #     # print 'peps opt 1', [m.shape for xx,m in np.ndenumerate(peps_dt_gl)],[m.shape for xx,m in np.ndenumerate(peps_opt_gl)], DMAX
            #    
            #     # print 'post opt'
            #     # print 'b_dir', b_dir, it
            #     # print 'opt', [(idx,m.shape) for idx,m in np.ndenumerate(sub_opt)]
            #     # print 'dt ', [(idx,m.shape) for idx,m in np.ndenumerate(sub_dt)]
            #     # print 'envs m', [[x.shape for x in m] for m in env_list_m]
            #     # print 'envs g', [[x.shape for x in m] for m in env_list_g]

            #     if regularize:    ## blows up if no regularization is done
            #         sub_opt = PEPX_GL.regularize(sub_opt)

            #     if build_env:
            #         sub_opt_norm = ENV_GL.embed_sites_norm_env(sub_opt, norm_env_m)
            #         sub_fidelity = ENV_GL.embed_sites_ovlp_env(np.conj(sub_dt),sub_opt,norm_env_g)/ sub_opt_norm
            #     else:
            #         sub_opt_norm = ENV_GL.embed_sites_norm(sub_opt, env_list_m)
	    #         sub_fidelity = ENV_GL.embed_sites_ovlp(np.conj(sub_opt),sub_dt,env_list_g)/ sub_opt_norm

            #         if np.isnan(sub_opt_norm):
            #             print 'nan opt norm!', sub_opt_norm

            #             print 'dt norm'
            #             print ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
            #            
            #             print 'init norm'
            #             if flatten:    sub_peps_gl = PEPX_GL.flatten(sub_peps_gl)
            #             print ENV_GL.embed_sites_norm(sub_peps_gl, env_list)

            #             raise RuntimeError('optimal norm is nan')

            #     # print 'fidelity', it, sub_it, sub_fidelity, sub_opt_norm

            #     if np.imag(sub_fidelity) > np.real(sub_fidelity):
            #         print sub_fidelity, sub_opt_norm
            #         print 'imag opt', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(sub_opt)]
            #         print 'real opt', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(sub_opt)]
            #         print 'imag dt',  [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(sub_dt)]
            #         print 'real dt',  [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(sub_dt)]
            #         if build_env:  print np.linalg.norm(norm_env_m), np.linalg.norm(norm_env_g)
            #         raise RuntimeWarning('imag fidelity or norm')


            #     not_converged = np.abs(sub_fidelity - sub_prev_fidelity) > 1.0e-10
            #     sub_fid_diff = sub_fidelity-sub_prev_fidelity

            #     if (sub_prev_fidelity > sub_fidelity+1.0e-12):
            #         sub_opt = sub_old_opt
            #         break

            #     # prev_alsq_diff = alsq_diff
            #     sub_prev_fidelity = sub_fidelity
            #     sub_old_opt       = sub_opt

            #     sub_it += 1

        # contracted b_dir to env; update other row/col
        if   b_dir == 'R':          peps_opt_gl[:,:1] = sub_opt
        elif b_dir == 'L':          peps_opt_gl[:,1:] = sub_opt
        elif b_dir == 'O':          peps_opt_gl[:1,:] = sub_opt
        elif b_dir == 'I':          peps_opt_gl[1:,:] = sub_opt
        else:   raise(ValueError),'b_dir not a valid direction'


        if regularize:    ## blows up if no regularization is done
            peps_opt_gl = PEPX_GL.regularize(peps_opt_gl)
            oldc_x, oldc_xx = None,None

        opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list)
	fidelity = ENV_GL.embed_sites_ovlp(np.conj(peps_opt_gl),peps_dt_gl,env_list)/opt_norm

        # opt_norm = ENV_GL.embed_sites_norm(sub_opt, env_list_m)
        # fidelity = ENV_GL.embed_sites_ovlp(np.conj(sub_opt), sub_dt, env_list_g)/opt_norm

        if np.isnan(opt_norm):
            print 'alsq_block_update 2x2 (2-site): nan opt norm!', opt_norm

            print 'dt norm'
            print ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
            
            print 'init norm'
            if flatten:    sub_peps_gl = PEPX_GL.flatten(sub_peps_gl)
            print ENV_GL.embed_sites_norm(sub_peps_gl, env_list)

            raise RuntimeError('optimal norm is nan')

        # print 'fidelity', it, fidelity, opt_norm

        if np.imag(fidelity) > np.real(fidelity):
            print fidelity, opt_norm
            print 'imag opt', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
            print 'real opt', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_opt_gl)]
            print 'imag dt',  [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
            print 'real dt',  [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(peps_dt_gl)]
            if build_env:  print np.linalg.norm(norm_env)
            raise RuntimeWarning('imag fidelity or norm')


        not_converged = np.abs(fidelity - prev_fidelity) > conv_tol #1.0e-6  #1.0e-8   #1.0e-10
        fid_diff = fidelity-prev_fidelity

        if (prev_fidelity > fidelity+1.0e-12):
            peps_opt_gl = old_peps_opt_gl
            break

        # prev_alsq_diff = alsq_diff
        prev_fidelity   = fidelity
        old_peps_opt_gl = peps_opt_gl

        it += 1

    if not_converged:  print 'not converged', it, fid_diff

    if normalize:
        peps_opt_gl = PEPX_GL.mul(1./opt_norm, peps_opt_gl)

    if flatten:
        peps_opt_gl = PEPX_GL.unflatten(peps_opt_gl,dbs)


    return peps_opt_gl


