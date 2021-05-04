import numpy as np
import time
import tens_fcts as tf

import MPX as MPX
import PEPX as PEPX
import PEPS_env as ENV





def meas_obs(peps,pepo,ind0=(0,0)):
    # operator acts on sites ind0 : ind0+len(mpo)
    L1,L2 = pepo.shape
    ix,iy = ind0

    peps_ = peps.copy()
    peps_[ix:ix+L1,iy:iy+L2] = dot(pepo,peps[ix:ix+L1,iy:iy+L2])

    expVal = PEPX.vdot(peps,peps_)
    return expVal

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
def simple_block_update(peps_list, connect_list, block1, DMAX=10, direction=0, regularize=True):
    ''' algorithm:  contract all sites into block and then do svd compression
                    not the most efficient, but maybe numerically more stable?
        pepo_part:  list of pepo sites that we're operating on
        block1:     nparray mpo-like block (1 x ... x 1) applied to sites on 'i' vertical bond
    '''    

    L = len(peps_list)

    if connect_list is None:  reorder = False
    else:                     reorder = True

    if isinstance(block1,MPX.MPX):   # this needs to go first bc MPX is also instance of np.ndarray
        L = len(peps_list)

        # print 'SU update', connect_list, [m.shape for m in peps_list]
        block_list, err = PEPX.mpo_update(peps_list, connect_list, block1, DMAX=DMAX, direction=direction, 
                                          regularize=regularize)

        # print 'SU update', [m.shape for m in block_list]


    elif isinstance(block1, np.ndarray):

        # ## check unitarity of trotter block (only unitary for real time evolution)
        # ns = (alt_block.ndim-2)/2
        # sqdim = int(np.sqrt(np.prod(alt_block.shape)))
        # block_io = alt_block.transpose(tf.site2io(ns)).reshape(sqdim,sqdim)

        # test = np.dot(np.conj(block_io.T), block_io)
        # print 'trotter block unitary err', np.linalg.norm(test-np.eye(sqdim))


        new_block = block1.copy()   
        if reorder:   pepsr_list, axT_invs = PEPX.connect_pepx_list(peps_list,connect_list,side='R')
        else:         pepsr_list = peps_list[:]

        peps1 = pepsr_list[0]
    
        new_block = np.einsum('AUD...,abcDe->AabcUe...',block1,peps1)
        new_block = tf.reshape(new_block,'ii,...')
    
        ind = 1
        while ind < len(peps_list):
        
            peps1 = pepsr_list[ind]
        
            p1 = 1 + ind*3 + 1
            p2 = p1 + 2
            p3 = p2 + 3 + (L-ind-1)*2 + 1
        
            indb = range(p1) + range(p2,p2+2) + range(p2+3,p3)
            inds = range(p1-1,p2) + range(p2+1,p2+3)
    
            # print new_block.shape, indb
            # print peps1.shape, inds
    
            new_block = np.einsum(new_block,indb,peps1,inds)
        
            ind += 1
        
        new_block = tf.reshape(new_block,'...,ii')
    
    
        # svd
        new_list, s_list = tf.decompose_block(new_block,L,direction,DMAX,svd_str=4,return_s=True)    # left canonical

        if regularize:
            new_list = PEPX.split_singular_vals(new_list,None,s_list,direction)

       
        # reorder indices
        if reorder:     block_list = PEPX.transpose_pepx_list(new_list,axT_invs)
        else:           block_list = new_list
    

    else:
        raise(TypeError), 'block1 should be np.ndarray (block) or MPX.MPX (mpo)'

    return block_list


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


def optimize_site(site_metric,site_grad):
    ''' return one optimized site
       (site_metric -- reshape into sq matrix split along bonds for the two missing sites)
       (vector
    '''

    site_shape = site_metric.shape
    sqdim = int(np.sqrt(np.prod(site_metric.shape)))
    M_block = site_metric.transpose(0,2,4,6,1,3,5,7).reshape(sqdim,sqdim)   # lLiIoOrR -> (lior)x(LIOR)
    G_vec = site_grad.reshape(sqdim,-1)    # (lior)xd

    M_inv = np.linalg.pinv(M_block, rcond=1.0e-8)
    site_ = np.matmul(M_inv,G_vec)

    # if np.linalg.norm(site_) > 5.0:
    #     print 'optimize site large norm', np.linalg.norm(site_), np.linalg.norm(G_vec),
    #     print np.linalg.norm(M_block), np.linalg.norm(M_inv)
    #     # exit()

    return site_.reshape(site_grad.shape)


def reduced_optimize_site(site_metric,site_grad,pepx_tens_u,pepx_tens_d,iso_leg):
    ''' return one optimized site
       (site_metric -- reshape into sq matrix split along bonds for the two missing sites)
       (vector
    '''

    ''' does QR decompositon of site at x_idx, contracts Q with env '''
    Qu,Ru,axT_inv = PEPX.QR_factor(pepx_tens_u,iso_leg)   # liox,xdr
    Qd,Rd,axT_inv = PEPX.QR_factor(pepx_tens_d,iso_leg)
    
    if   iso_leg=='l':  grad_qo = np.einsum('liord,iorx->xld',site_grad,Qu)
    elif iso_leg=='i':  grad_qo = np.einsum('liord,lorx->xid',site_grad,Qu)
    elif iso_leg=='o':  grad_qo = np.einsum('liord,lirx->xod',site_grad,Qu)
    elif iso_leg=='r':  grad_qo = np.einsum('liord,liox->xrd',site_grad,Qu)
    
    if iso_leg=='l':
        metric_qx = np.einsum('lLiIoOrR,iorx->lLIORx',site_metric,Qu)
        metric_qq = np.einsum('lLIORx,IORX->xlXL',metric_qx,Qd)
    elif iso_leg=='i':
        metric_qx = np.einsum('lLiIoOrR,lorx->LiIORx',site_metric,Qu)
        metric_qq = np.einsum('LiIORx,LORX->xiXI',metric_qx,Qd)
    elif iso_leg=='o':
        metric_qx = np.einsum('lLiIoOrR,lirx->LIoORx',site_metric,Qu)
        metric_qq = np.einsum('LIoORx,LIRX->xoXO',metric_qx,Qd)
    elif iso_leg=='r':
        metric_qx = np.einsum('lLiIoOrR,liox->LIOrRx',site_metric,Qu)
        metric_qq = np.einsum('LIOrRx,LIOX->xrXR',metric_qx,Qd)
    
    sqdim = int(np.sqrt(np.prod(metric_qq.shape)))
    M_block = metric_qq.reshape(sqdim,sqdim)   # xrXR -> (xr)*(XR)
    G_vec = grad_qo.reshape(sqdim,-1)          # (xr)*(d)

    M_inv = np.linalg.pinv(M_block, rcond=1.0e-8)
    Rd_ = np.matmul(M_inv,G_vec).reshape(Rd.transpose(0,2,1).shape)   # XRd
    Rd_ = Rd_.transpose(0,2,1)   # XdR

    site_ = PEPX.QR_contract(Qd,Rd_,axT_inv)

    return site_




# @profile
def alsq_block_update(sub_peps,connect_list,inds_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                      init_guess=None, gauge_env=False,ensure_pos=False,build_env=True):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        inds_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over
        build_env:  True if build env and then reuse it
    '''

    L1,L2 = sub_peps.shape
    xs,ys = inds_list
    # print 'alsq x, y', xs,ys

    if build_env: 
        ensure_pos=True
        norm_env = ENV.build_env(env_list,ensure_pos=ensure_pos)

        if gauge_env:
            norm_env = ENV.gauge_fix_env(norm_env,np.conj(sub_peps),np.conj(sub_peps),connect_list)

    peps_list = sub_peps[xs,ys]
    peps_dt = PEPX.copy(sub_peps)
    peps_su = PEPX.copy(sub_peps)

    # print 'alsq init', [m.shape for m in peps_list]

    # apply trotter and then svd
    list_dt = simple_block_update(peps_list,connect_list,trotter_block,DMAX=-1,direction=0,regularize=True)
    peps_dt[xs,ys]  = list_dt


    # # perform ITE normalization here
    if build_env:        dt_norm = ENV.embed_sites_norm_env(peps_dt, norm_env)
    else:                dt_norm = ENV.embed_sites_norm(peps_dt, env_list)
    peps_dt = PEPX.mul(1./dt_norm, peps_dt)
    

    if isinstance(init_guess,np.ndarray): 
         # init guess + some noise to increase bond dimension
         leg_ind = PEPX.leg2ind(connect_list[0])
         if sub_peps[0,0].shape[leg_ind] < DMAX:

             list_su, errs = PEPX.compress_peps_list( peps_dt[xs,ys], connect_list, DMAX, direction=0,regularize=True)
             peps_su[xs,ys] = list_su
             peps_opt = peps_su

         else:
             peps_opt = init_guess

    elif init_guess in ['rand','random']:
         leg_ind_o = [PEPX.leg2ind(c_leg) for c_leg in connect_list]+[None]
         leg_ind_i = [None]+[PEPX.leg2ind(PEPX.opposite_leg(c_leg)) for c_leg in connect_list]
         shape_array = np.empty((L1,L2),dtype=tuple)
         for i in range(len(peps_list)):
             ind = (xs[i],ys[i])
             rand_shape = list(peps_list[i].shape)
             if leg_ind_o[i] is not None:     rand_shape[leg_ind_o[i]] = DMAX
             if leg_ind_i[i] is not None:     rand_shape[leg_ind_i[i]] = DMAX
             shape_array[ind] = tuple(rand_shape)

         norm_env = ENV.build_env(env_list,ensure_pos=ensure_pos)
         peps_opt = ENV.random_tens_env( shape_array,env=norm_env )
         list_opt, errs = PEPX.compress_peps_list( peps_opt[xs,ys], connect_list, DMAX, direction=0,regularize=True)
         peps_opt[xs,ys] = list_opt

    else:     # init_guess is None or init_guess in ['su','SU','svd']:
         list_su, errs = PEPX.compress_peps_list( peps_dt[xs,ys], connect_list, DMAX, direction=0,regularize=True)
         peps_su[xs,ys] = list_su
         peps_opt = peps_su

    old_peps_opt = peps_opt


    if site_idx is None:        site_idx = [m for m in np.ndindex(sub_peps.shape)]
    else:                       site_idx = [m for m in site_idx]


    not_converged = True
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0
    while (not_converged or it < 3) and it < 500:

        idx_ind = 0
        for idx in site_idx:
            # env_x  = ENV.embed_sites_xo(np.conj(peps_opt),peps_dt ,env_list,idx)
            # env_xx = ENV.embed_sites_xx(np.conj(peps_opt),peps_opt,env_list,idx)
            # env_x  = ENV.embed_sites_xo_bm(np.conj(peps_opt),peps_dt,env_list,idx,XMAX=XMAX,update_conn=connect_list)
            # env_xx = ENV.embed_sites_xx(np.conj(peps_opt),peps_opt,env_list,idx)#,XMAX=XMAX,update_conn=connect_list)

            if build_env:
                env_x  = ENV.embed_sites_xo_env(np.conj(peps_opt),peps_dt, norm_env,idx)
                env_xx = ENV.embed_sites_xx_env(np.conj(peps_opt),peps_opt,norm_env,idx)
            else:
                env_x  = ENV.embed_sites_xo(np.conj(peps_opt),peps_dt ,env_list,idx)
                env_xx = ENV.embed_sites_xx(np.conj(peps_opt),peps_opt,env_list,idx)


            # site_opt = optimize_site(np.real(env_xx),np.real(env_x))
            site_opt = optimize_site(env_xx,env_x)
            # print 'site opt', np.linalg.norm(site_opt), np.linalg.norm(env_xx), np.linalg.norm(env_x)
           
            # print 'site opt', peps_opt[idx].shape, site_opt.shape
            peps_opt[idx] = site_opt

            if np.any( [np.linalg.norm(m) > 10 for x,m in np.ndenumerate(peps_opt)] ):
                print 'WARNING: optimal peps large norm', idx, [np.linalg.norm(m) for x,m in np.ndenumerate(peps_opt)]
                # print 'env norms', np.linalg.norm(env_x), np.linalg.norm(env_xx), np.linalg.norm(norm_env)

            # peps_opt = PEPX.regularize(peps_opt,idx)
     
        ############
        reg_opt, errs = PEPX.compress_peps_list(peps_opt[xs,ys],connect_list,-1,regularize=True)
        peps_opt[xs,ys] = reg_opt

        if build_env:     
            opt_norm = ENV.embed_sites_norm_env(peps_opt, norm_env)
            fidelity = ENV.embed_sites_ovlp_env(np.conj(peps_dt),peps_opt,norm_env)/ opt_norm
        else:             
            opt_norm = ENV.embed_sites_norm(peps_opt, env_list, XMAX=XMAX)
            fidelity = ENV.embed_sites_ovlp(np.conj(peps_dt),peps_opt,env_list)/ opt_norm

        if np.imag(fidelity) > np.real(fidelity):
            print 'imag diff', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(diff_state)]
            print 'real diff', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(diff_state)]
            norm_env = ENV.build_env(env_list,ensure_pos=ensure_pos)
            print np.linalg.norm(norm_env)

        try:
            # not_converged = np.any( np.diff(fidelities[-5:]) > 1.0e-5 )
            # not_converged = np.abs(alsq_diff) > 5.0e-5
            # not_converged = (np.abs(prev_alsq_diff) - np.abs(alsq_diff)) > 1.0e-8   # alsq decreases
            not_converged = (fidelity - prev_fidelity ) > 1.0e-6  # 1.0e-8

            if (np.abs(prev_fidelity)>np.abs(fidelity)+1.0e-12):
                peps_opt = old_peps_opt
                print 'fidelity went down or alsq diff went up'
                break

            # prev_alsq_diff = alsq_diff
            prev_fidelity  = fidelity
            # old_peps_opt   = peps_opt

        except(IndexError):   pass

        it += 1

    # if not_converged:   print 'not converged', it, prev_alsq_diff - alsq_diff, fidelity - prev_fidelity
    if not_converged:   print 'not converged', it, fidelity - prev_fidelity

    return PEPX.mul(1./opt_norm, peps_opt)



def red_alsq_block_update(sub_peps,connect_list,inds_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                          init_guess=None, gauge_env=False,ensure_pos=False,build_env=True, normalize=False):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        inds_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over
        build_env:  True if build env and then reuse it
    '''

    L1,L2 = sub_peps.shape
    xs,ys = inds_list
    # print 'alsq x, y', xs,ys

    if build_env: 
        ensure_pos=True
        norm_env = ENV.build_env(env_list,ensure_pos=ensure_pos)

        if gauge_env:
            norm_env = ENV.gauge_fix_env(norm_env,np.conj(sub_peps),np.conj(sub_peps),connect_list)

    peps_list = sub_peps[xs,ys]
    peps_dt = PEPX.copy(sub_peps)
    peps_su = PEPX.copy(sub_peps)

    # print 'alsq init', [m.shape for m in peps_list]

    # apply trotter and then svd
    list_dt = simple_block_update(peps_list,connect_list,trotter_block,DMAX=-1,direction=0,regularize=True)
    peps_dt[xs,ys]  = list_dt


    # # perform ITE normalization here
    if build_env:        dt_norm = ENV.embed_sites_norm_env(peps_dt, norm_env)
    else:                dt_norm = ENV.embed_sites_norm(peps_dt, env_list)
    peps_dt = PEPX.mul(1./dt_norm, peps_dt)
    

    if isinstance(init_guess,np.ndarray): 
         # init guess + some noise to increase bond dimension
         leg_ind = PEPX.leg2ind(connect_list[0])
         if sub_peps[0,0].shape[leg_ind] < DMAX:

             list_su, errs = PEPX.compress_peps_list( peps_dt[xs,ys], connect_list, DMAX, direction=0,regularize=True)
             peps_su[xs,ys] = list_su
             peps_opt = peps_su

         else:
             peps_opt = init_guess

    elif init_guess in ['rand','random']:
         leg_ind_o = [PEPX.leg2ind(c_leg) for c_leg in connect_list]+[None]
         leg_ind_i = [None]+[PEPX.leg2ind(PEPX.opposite_leg(c_leg)) for c_leg in connect_list]
         shape_array = np.empty((L1,L2),dtype=tuple)
         for i in range(len(peps_list)):
             ind = (xs[i],ys[i])
             rand_shape = list(peps_list[i].shape)
             if leg_ind_o[i] is not None:     rand_shape[leg_ind_o[i]] = DMAX
             if leg_ind_i[i] is not None:     rand_shape[leg_ind_i[i]] = DMAX
             shape_array[ind] = tuple(rand_shape)

         raise(NotImplementedError), 'random init state for alsq not implemented'

         # norm_env = ENV.build_env(env_list,ensure_pos=ensure_pos)
         # peps_opt = ENV.random_tens_env( shape_array,env=norm_env )
         # list_opt, errs = PEPX.compress_peps_list( peps_opt[xs,ys], connect_list, DMAX, direction=0,regularize=True)
         # peps_opt[xs,ys] = list_opt

    else:     # init_guess is None or init_guess in ['su','SU','svd']:
         list_su, errs = PEPX.compress_peps_list( peps_dt[xs,ys], connect_list, DMAX, direction=0,regularize=True)
         peps_su[xs,ys] = list_su
         peps_opt = peps_su

    old_peps_opt = peps_opt


    # if site_idx is None:        site_idx = [m for m in np.ndindex(sub_peps.shape)]
    # else:                       site_idx = [m for m in site_idx]

    if site_idx is None:
        if (L1,L2) == (2,2):     site_idx = [(0,0),(0,1),(1,1),(1,0)]
        else:                    site_idx = [m for m in np.ndindex(sub_peps.shape)]
    else:                        site_idx = [m for m in site_idx]

    temp_xs = [m[0] for m in site_idx]
    temp_ys = [m[1] for m in site_idx]
    loop_conn = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)

    not_converged = True
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0
    while (not_converged or it < 3) and it < 500:

        idx_ind = 0

        for idx in site_idx:

            # env_x  = ENV.embed_sites_xo(np.conj(peps_opt),peps_dt ,env_list,idx)
            # env_xx = ENV.embed_sites_xx(np.conj(peps_opt),peps_opt,env_list,idx)
            # env_x  = ENV.embed_sites_xo_bm(np.conj(peps_opt),peps_dt,env_list,idx,XMAX=XMAX,update_conn=connect_list)
            # env_xx = ENV.embed_sites_xx(np.conj(peps_opt),peps_opt,env_list,idx)#,XMAX=XMAX,update_conn=connect_list)

            if build_env:
                env_x  = ENV.embed_sites_xo_env(np.conj(peps_opt),peps_dt, norm_env,idx)
                env_xx = ENV.embed_sites_xx_env(np.conj(peps_opt),peps_opt,norm_env,idx)
            else:
                env_x  = ENV.embed_sites_xo(np.conj(peps_opt),peps_dt ,env_list,idx)
                env_xx = ENV.embed_sites_xx(np.conj(peps_opt),peps_opt,env_list,idx)


            iso_leg = loop_conn[idx_ind]
            site_opt = reduced_optimize_site(env_xx,env_x,np.conj(peps_opt[idx]),peps_dt[opt],iso_leg)
            # print 'site opt', np.linalg.norm(site_opt), np.linalg.norm(env_xx), np.linalg.norm(env_x)
           
            # print 'site opt', peps_opt[idx].shape, site_opt.shape
            peps_opt[idx] = site_opt

            if np.any( [np.linalg.norm(m) > 10 for x,m in np.ndenumerate(peps_opt)] ):
                print 'WARNING: optimal peps large norm', idx, [np.linalg.norm(m) for x,m in np.ndenumerate(peps_opt)]
                # print 'env norms', np.linalg.norm(env_x), np.linalg.norm(env_xx), np.linalg.norm(norm_env)

            # peps_opt = PEPX.regularize(peps_opt,idx)

            idx_ind += 1
     
        ############
        reg_opt, errs = PEPX.compress_peps_list(peps_opt[xs,ys],connect_list,-1,regularize=True)
        peps_opt[xs,ys] = reg_opt

        if build_env:     
            opt_norm = ENV.embed_sites_norm_env(peps_opt, norm_env)
            fidelity = ENV.embed_sites_ovlp_env(np.conj(peps_dt),peps_opt,norm_env)/ opt_norm
        else:             
            opt_norm = ENV.embed_sites_norm(peps_opt, env_list, XMAX=XMAX)
            fidelity = ENV.embed_sites_ovlp(np.conj(peps_dt),peps_opt,env_list)/ opt_norm

        if np.imag(fidelity) > np.real(fidelity):
            print 'imag diff', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(diff_state)]
            print 'real diff', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(diff_state)]
            norm_env = ENV.build_env(env_list,ensure_pos=ensure_pos)
            print np.linalg.norm(norm_env)

        try:
            # not_converged = np.any( np.diff(fidelities[-5:]) > 1.0e-5 )
            # not_converged = np.abs(alsq_diff) > 5.0e-5
            # not_converged = (np.abs(prev_alsq_diff) - np.abs(alsq_diff)) > 1.0e-8   # alsq decreases
            not_converged = (fidelity - prev_fidelity ) > 1.0e-6  # 1.0e-8

            if (np.abs(prev_fidelity)>np.abs(fidelity)+1.0e-12):
                peps_opt = old_peps_opt
                print 'fidelity went down or alsq diff went up'
                break

            # prev_alsq_diff = alsq_diff
            prev_fidelity  = fidelity
            # old_peps_opt   = peps_opt

        except(IndexError):   pass

        it += 1

    # if not_converged:   print 'not converged', it, prev_alsq_diff - alsq_diff, fidelity - prev_fidelity
    if not_converged:   print 'not converged', it, fidelity - prev_fidelity

    return PEPX.mul(1./opt_norm, peps_opt)


