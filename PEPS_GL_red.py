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
import PEPS_GL as ALSQ
import PEPS_env as ENV
import PEPS_GL_env_nolam as ENV_GL
import TLDM_GL_env_nolam as ENV_DM



def alsq_block_update_red_2x2(sub_peps_gl,connect_list,inds_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                          init_guess=None, ensure_pos=False, normalize=False, regularize=True, build_env=False,
                          qr_reduce=False, flatten=False):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        inds_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over
    '''

    L1,L2 = sub_peps_gl.shape
    xs,ys = inds_list

    ind0 = (inds_list[0][0],inds_list[1][0])
    gam_list, lam_list, axT_invs = PEPX_GL.get_sites(sub_peps_gl,ind0,connect_list)

    # apply trotter and then svd
    gam_dt, lam_dt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=-1,direction=0,normalize=normalize)
    peps_dt_gl = PEPX_GL.set_sites(sub_peps_gl,ind0,connect_list,gam_dt,lam_dt,axT_invs) 

    if flatten:
        dbs = peps_dt_gl.phys_bonds
        peps_dt_gl  = PEPX_GL.flatten(peps_dt_gl)
        # peps_opt_gl = PEPX_GL.flatten(peps_opt_gl)
        # print 'flattened', [m.shape for idx,m in np.ndenumerate(peps_dt_gl)]

    if normalize:
        if build_env:        dt_norm = ENV_GL.embed_sites_norm_env(peps_dt_gl, norm_env)   # env is without lambdas
        else:                dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list)
        # print 'dt norm', dt_norm
        peps_dt_gl = PEPX_GL.mul(1./dt_norm, peps_dt_gl)

    if isinstance(init_guess,np.ndarray): 
         # init guess -- increase bond dimension using svd init guess
         leg_ind = PEPX.leg2ind(connect_list[0])
         if sub_peps_gl[0,0].shape[leg_ind] < DMAX:
             peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)
         else:
             peps_opt_gl = init_guess

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
         peps_opt_gl = PEPX_GL.compress(peps_dt_gl,DMAX)

    old_peps_opt_gl = peps_opt_gl


    ## define order of sites in ALSQ optimization
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
            loop_conn = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=False)
            conns += [loop_conn + [PEPX_GL.opposite_leg(loop_conn[-1])]]

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
            conns += [loop_conn + [PEPX_GL.opposite_leg(loop_conn[-1])]]
    

        
    not_converged = True
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0

    reuse_c = True
    oldc_x, oldc_xx = [None, None]   # remembering old corners to skip extra computations
    while (not_converged or it < 3) and it < 500:

        for xi in range(len(indxs)):

            site_list = indxs[xi]
            site_conn = conns[xi]

            ''' !!! with this method still don't avoid the contraction costs (extra scaling hidden
                in larger bond dimension of modified env
                however, this method might give better convergence?
                one could also build env and then ensure positivity (but would change every it1
            '''
            # first and last element in list -- QR optimization

            for xj in range(len(site_list)):

                if xj == 0 or xj == len(site_list)-1:

                    iso_leg = site_conn[xj]

                    env_xx,qu_idx,qd_idx,axT_inv,oldc_xx = \
                           ENV_GL.red_embed_sites_xx(np.conj(peps_opt_gl),peps_opt_gl,env_list,idx,iso_leg,
                                                     old_corners=oldc_xx,return_corners=True)
                    env_x,oldc_x = \
                           ENV_GL.red_embed_sites_xo(np.conj(peps_opt_gl),peps_dt_gl ,env_list,idx,iso_leg,qu_idx,qd_idx,
                                                     old_corners=oldc_x,return_corners=True)
                    r_opt = red_optimize_site(env_xx,env_x,ensure_pos=ensure_pos)   # includes lams
                    # r_opt = optimize_site_constrained(env_xx,env_x,do_reduced=qr_reduce)
                    site_opt = PEPX.QR_contract(qd_idx,r_opt,axT_inv,d_end=True)

                else:
                    env_x , oldc_x  = ENV_GL.embed_sites_xo(np.conj(peps_opt_gl),peps_dt_gl ,env_list,idx,
                                                         old_corners=oldc_x, return_corners=True)
                    env_xx, oldc_xx = ENV_GL.embed_sites_xx(np.conj(peps_opt_gl),peps_opt_gl,env_list,idx,
                                                         old_corners=oldc_xx,return_corners=True)

                    site_opt = optimize_site(env_xx,env_x,ensure_pos=ensure_pos)   # includes lams
                    # site_opt = optimize_site_2(env_xx,env_x,do_lsq=True)   # includes lams
                    # site_opt = optimize_site_constrained(env_xx,env_x)   # includes lams
                    # site_opt = optimize_site_constrained_2(env_xx,env_x)   # includes lams
                    # site_opt = optimize_site_constrained_3(env_xx,env_x)   # includes lams


                ## only change gamma (keep lambda from dt svd)
                # peps_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,peps_opt_gl.lambdas[idx])

                if not reuse_c:   oldc_x, oldc_xx = None,None

                ## do svd on site_opt + next site to keep it close to GL form
                peps_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,peps_opt_gl.lambdas[idx])
                gs, ls, axTs = PEPX_GL.get_sites(peps_opt_gl,idx,iso_leg)
                gs_, ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,normalize=False)
                peps_opt_gl = PEPX_GL.set_sites(peps_opt_gl,idx,iso_leg,gs_,ls_,axTs)

                # denote which site is modified due to canonicalization (old corner cannot be reused)
                # only is affected when above setting of sites is not the same direction as next optimization step
                # eg. when loop direction is changed a
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

                idx_ind += 1

                if np.any( [np.linalg.norm(peps_opt_gl.lambdas[xx]) > 10 for xx in np.ndindex(L1,L2,4)] ):
                    print 'large lambda', idx, iso_leg, [np.linalg.norm(peps_opt_gl.lambdas[xx]) for xx in np.ndindex(L1,L2,4)]
                    print 'large lambda', idx, iso_leg, [np.linalg.norm(peps_dt_gl.lambdas[xx]) for xx in np.ndindex(L1,L2,4)]
                    print [np.linalg.norm(m) for idx, m in np.ndenumerate(peps_opt_gl)]
                    # print 'env norms', np.linalg.norm(env_x), np.linalg.norm(env_xx), np.linalg.norm(norm_env)
                    opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list, old_corners=None)
                    dt_norm = ENV_GL.embed_sites_norm(peps_dt_gl, env_list, old_corners=None)
                    print opt_norm, dt_norm
                    raise RuntimeWarning('large lambda, '+str(idx))

        ############

        if regularize:    ## blows up if no regularization is done
            peps_opt_gl = PEPX_GL.regularize(peps_opt_gl)
            oldc_x, oldc_xx = None,None
            # print 'regularize', [np.linalg.norm(m) for m in gam_opt], lam_opt

        opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list, old_corners=oldc_xx)
	fidelity = ENV_GL.embed_sites_ovlp(np.conj(peps_opt_gl),peps_dt_gl,env_list,old_corners=oldc_x)/opt_norm
        # opt_norm = ENV_GL.embed_sites_norm(peps_opt_gl, env_list, old_corners=None)
	# fidelity = ENV_GL.embed_sites_ovlp(np.conj(peps_opt_gl),peps_dt_gl,env_list,old_corners=None)/ opt_norm

        # print 'opt norm', opt_norm

        # for env in env_list:
        #     if len(env) == 2:
        #         print env[0].shape, env[1].shape
        #         b = np.tensordot(env[0],env[1],axes=(-1,0))
        #         sqdim = int(np.sqrt(np.prod(b.shape)))
        #         B = b.transpose(0,1,3,2,4,5).reshape(sqdim,sqdim)
        #     elif len(env) == 1:
        #         print env[0].shape
        #         sqdim = int(np.sqrt(np.prod(env[0].shape)))
        #         B = env[0].reshape(sqdim,sqdim)

        #     print 'env H?', np.allclose(B,np.conj(B.T))
                
        if np.isnan(opt_norm):
            print 'nan opt norm!', opt_norm

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


        not_converged = np.abs(fidelity - prev_fidelity) > 1.0e-8
        fid_diff = fidelity-prev_fidelity

        if (prev_fidelity > fidelity+1.0e-12):
            peps_opt_gl = old_peps_opt_gl
            oldc_x, oldc_xx = None, None
            print 'fidelity went down or alsq diff went up, reset opt gl'
            break

        # prev_alsq_diff = alsq_diff
        prev_fidelity     = fidelity
        old_peps_opt_gl   = peps_opt_gl

        # try:
        #     # not_converged = np.any( np.diff(fidelities[-5:]) > 1.0e-5 )
        #     # not_converged = np.abs(alsq_diff) > 5.0e-5
        #     # not_converged = (np.abs(prev_alsq_diff) - np.abs(alsq_diff)) > 1.0e-8   # alsq decreases
        #     not_converged = np.abs(fidelity - prev_fidelity) > 1.0e-10

        #     if (prev_fidelity > fidelity+1.0e-12):
        #         peps_opt_gl = old_peps_opt_gl
        #         print 'fidelity went down or alsq diff went up'
        #         break

        #     # prev_alsq_diff = alsq_diff
        #     prev_fidelity     = fidelity
        #     old_peps_opt_gl   = peps_opt_gl

        # except(IndexError):   pass

        it += 1

    if not_converged:  print 'not converged', it, fid_diff

    if normalize:
        peps_opt_gl = PEPX_GL.mul(1./opt_norm, peps_opt_gl)

    # print 'alsq norm', ENV_GL.embed_sites_norm(peps_opt_gl, env_list)

    if flatten:
        peps_opt_gl = PEPX_GL.unflatten(peps_opt_gl,dbs)


    return peps_opt_gl


def new_Q_envs_L(sub_peps_u,sub_peps_d,env_list):

    bL,bI,bO,bR = env_list

    tens00_u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2,3])
    tens00_d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2,3])
    tens10_u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[3])
    tens10_d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[3])

    qu00, ru00, axT00_inv = PEPX.QR_factor(tens00_u,3,d_end=True)  # liox xrd
    qd00, rd00, axT00_inv = PEPX.QR_factor(tens00_d,3,d_end=True)  # liox xrd
    qu10, ru10, axT10_inv = PEPX.QR_factor(tens10_u,3,d_end=True)  
    qd10, rd10, axT10_inv = PEPX.QR_factor(tens10_d,3,d_end=True)  

    bL_0 = np.einsum('xlLy,xiIw->wlLiIy',  bL[0],bI[0])
    bL_0 = np.einsum('wlLiIy,lioq->woqLIy',bL_0,qu00)
    bL_0 = np.einsum('woqLIy,LIOQ->wqQoOy',bL_0,qd00)
    bL_0 = tf.reshape(bL_0,'i,i,i,iii')

    bL_1 = np.einsum('wlLx,xoOy->wlLoOy',  bL[1],bO[0])
    bL_1 = np.einsum('wlLoOy,lioq->wiqLOy',bL_1,qu10)
    bL_1 = np.einsum('wiqLOy,LIOQ->wiIqQy',bL_1,qd10)
    bL_1 = tf.reshape(bL_1,'iii,i,i,i')

    env_list_new = [[bL_0,bL_1],bI[1:],bO[1:],bR[:]]

    sub_new_u = sub_peps_u.copy()
    sub_new_d = sub_peps_d.copy()
    sub_new_u[0,0] = (qu00,ru00)
    sub_new_d[0,0] = (qd00,rd00)
    sub_new_u[1,0] = (qu10,ru10)
    sub_new_d[1,0] = (qd10,rd10)

    return env_list_new, sub_new_u, sub_new_d, axT00_inv, axT10_inv


def new_Q_envs_I(sub_peps_u,sub_peps_d,,env_list):

    bL,bI,bO,bR = env_list

    tens00_u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2,3])
    tens00_d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2,3])
    tens01_u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[2])
    tens01_d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[2])

    qu00, ru00, axT00_inv = PEPX.QR_factor(tens00_u,2,d_end=True)  # liox xrd
    qd00, rd00, axT00_inv = PEPX.QR_factor(tens00_d,2,d_end=True)  # liox xrd
    qu01, ru01, axT01_inv = PEPX.QR_factor(tens01_u,2,d_end=True)  
    qd01, rd01, axT01_inv = PEPX.QR_factor(tens01_d,2,d_end=True)  

    bI_0 = np.einsum('xlLy,xiIw->wlLiIy',  bL[0],bI[0])
    bI_0 = np.einsum('wlLiIy,lirq->wrqLIy',bI_0,q0u)
    bI_0 = np.einsum('wrqLIy,LIRQ->wqQrRy',bI_0,q0d)
    bI_0 = tf.reshape(bL_0,'i,i,i,iii')

    bI_1 = np.einsum('wiIx,xrRy->wiIrRy',  bI[1],bR[0])
    bI_1 = np.einsum('wiIrRy,lirq->wlqIRy',bI_1,q1u)
    bI_1 = np.einsum('wlqIRy,LIRQ->wlLqQy',bI_1,q1d)
    bI_1 = tf.reshape(bL_1,'iii,i,i,i')

    env_list_xx = [bL[1:],[bI_0,bI_1],bO[:],bR[1:]]

    sub_new_u = sub_peps_u.copy()
    sub_new_d = sub_peps_d.copy()
    sub_new_u[0,0] = (qu00,ru00)
    sub_new_d[0,0] = (qd00,rd00)
    sub_new_u[0,1] = (qu01,ru01)
    sub_new_d[0,1] = (qd01,rd01)

    return env_list_new, sub_new_u, sub_new_d, axT00_inv, axT01_inv


def new_Q_envs_R(sub_peps_u,sub_peps_d,env_list):

    bL,bI,bO,bR = env_list

    tens01_u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0,2])
    tens01_d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0,2])
    tens11_u = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[0])
    tens11_d = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[0])

    qu01, ru01, axT01_inv = PEPX.QR_factor(tens01_u,0,d_end=True)  # liox xrd
    qd01, rd01, axT01_inv = PEPX.QR_factor(tens01_d,0,d_end=True)  # liox xrd
    qu11, ru11, axT11_inv = PEPX.QR_factor(tens11_u,0,d_end=True)  
    qd11, rd11, axT11_inv = PEPX.QR_factor(tens11_d,0,d_end=True)  

    bR_0 = np.einsum('wiIx,xrRy->wiIrRy',  bI[1],bR[0])
    bR_0 = np.einsum('wiIrRy,iorq->wIRoqy',bR_0,q0u)
    bR_0 = np.einsum('wIRoqy,IORQ->wqQoOy',bR_0,q0d)
    bR_0 = tf.reshape(bL_0,'i,i,i,iii')

    bR_1 = np.einsum('yoOx,xrRw->woOrRy',  bO[1],bR[1])
    bR_1 = np.einsum('woOrRy,iorq->wiqORy',bR_1,q1u)
    bR_1 = np.einsum('wiqORy,IORQ->wiIqQy',bR_1,q1d)
    bR_1 = tf.reshape(bL_1,'iii,i,i,i')

    env_list_new = [bL[:],bI[:1],bO[:1],[bR_0,bR_1]]

    sub_new_u = sub_peps_u.copy()
    sub_new_d = sub_peps_d.copy()
    sub_new_u[0,1] = (qu01,ru01)
    sub_new_d[0,1] = (qd01,rd01)
    sub_new_u[1,1] = (qu11,ru11)
    sub_new_d[1,1] = (qd11,rd11)

    return env_list_new, sub_new_u, sub_new_d, axT01_inv, axT11_inv


def new_Q_envs_O(q0u,q0d,q1u,q1d,env_list):

    bL,bI,bO,bR = env_list

    tens10_u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[0,1])
    tens10_d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[0,1])
    tens11_u = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[1])
    tens11_d = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[1])

    qu10, ru10, axT10_inv = PEPX.QR_factor(tens10_u,1,d_end=True)  # liox xrd
    qd10, rd10, axT10_inv = PEPX.QR_factor(tens10_d,1,d_end=True)  # liox xrd
    qu11, ru11, axT11_inv = PEPX.QR_factor(tens11_u,1,d_end=True)  
    qd11, rd11, axT11_inv = PEPX.QR_factor(tens11_d,1,d_end=True)  

    bO_0 = np.einsum('wlLx,xoOy->wlLoOy',  bL[1],bO[0])
    bO_0 = np.einsum('wlLoOy,lorq->wLOrqy',bO_0,q0u)
    bO_0 = np.einsum('wLOrqy,LORQ->wqQrRy',bO_0,q0d)
    bO_0 = tf.reshape(bL_0,'i,i,i,iii')

    bO_1 = np.einsum('woOx,xrRy->woOrRy',  bO[1],bR[1])
    bO_1 = np.einsum('woOrRy,lorq->wlqORy',bO_1,q1u)
    bO_1 = np.einsum('wlqORy,LORQ->wlLqQy',bO_1,q1d)
    bO_1 = tf.reshape(bL_1,'iii,i,i,i')

    env_list_xx = [bL[:1],bI[:],[bO_0,bO_1],bR[:1]]

    sub_new_u = sub_peps_u.copy()
    sub_new_d = sub_peps_d.copy()
    sub_new_u[1,0] = (qu10,ru10)
    sub_new_d[1,0] = (qd10,rd10)
    sub_new_u[1,1] = (qu11,ru11)
    sub_new_d[1,1] = (qd11,rd11)

    return env_list_new, sub_new_u, sub_new_d, axT10_inv, axT11_inv



def red_embed_sites(sub_peps_u, sub_peps_d, rs_u, rs_d, envs, x_idx, idx_key):

    bL, bI, bO, bR = envs
    L1,L2 = sub_peps_u.shape

    tens_u_idx = PEPX_GL.get_site(sub_peps_u,x_idx)
    tens_d_idx = PEPX_GL.get_site(sub_peps_d,x_idx)

    if (L1,L2) == (1,2):   # horizontal trotter step

        if x_idx == (0,0):

            s2u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0])

            ## update site 2
            s2u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0])
            s2d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0])
    
            env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
            env_block2 = np.einsum('wiIrRy,liord->wlIoRdy',env_block2,s2u)
            env_block2 = np.einsum('wlIoRdy,LIORd->wlLoOy',env_block2,s2d)
            env_block2 = np.einsum('wlLoOy,zoOy->wlLz',env_block2,bO[1])
    
            ## site 1 boundary
            # assume iso_leg = 'r'
            if idx_key == 'xx':    ## metric
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lioq->wLIoqy',env_block1,qu_idx)
                env_block1 = np.einsum('wLIoqy,LIOQ->woOqQy',env_block1,qd_idx)
                env_block1 = np.einsum('yoOz,woOqQy->wqQz',bO[0],env_block1)
                env_out = np.einsum('wqQz,wrRz->qQrR',env_block1,env_block2)
            elif idx_key == 'xo':  ## grad (missing bra)
                s1d = PEPX_GL.get_site(sub_peps_d,(0,0))
                env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lioq->wLIoqy',env_block1,qu_idx)
                env_block1 = np.einsum('wLIoqy,LIORd->woORqdy',env_block1,s1d)
                env_block1 = np.einsum('woORqdy,yoOz->wRqdz',env_block1,bO[0])
                env_out = np.einsum('wRqdz,wrRz->qrd',env_block1,env_block2)
    

        elif x_idx == (0,1):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[3])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[3])
        
            # env_block1 = np.einsum('xiIw,liord->wlIorxd',bI[0],s1u)
            # env_block1 = np.einsum('wlIorxd,LIORd->wlLoOrRx',env_block1,s1d)     # inner boundary
            # env_block1 = np.einsum('wlLoOrRx,xlLy->woOrRy',env_block1,bL[0])     # left boundary
            # env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])         # outer boundary
            env_block1 = np.einsum('xlLy,xiIw->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)
            env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
            env_block1 = np.einsum('woOrRy,yoOz->wrRz',env_block1,bO[0])

            ## site 2 boundary
            # assume iso_leg = 'l'
            if idx_key == 'xx':
                env_block2 = np.einsum('xiIy,yrRz->xiIrRz',bI[1],bR[0])
                env_block2 = np.einsum('xiIrRz,iorq->xqIoRz',env_block2,qu_idx)
                env_block2 = np.einsum('xqIoRz,IORQ->xqQoOz',env_block2,qd_idx)
                env_block2 = np.einsum('xqQoOy,zoOy->xqQz',env_block2,bO[1])
                env_out = np.einsum('wlLz,wqQz->qQlL',env_block1,env_block2)
            elif idx_key == 'xo':
                s2d = PEPX_GL.get_site(sub_peps_d,(0,1))
                env_block2 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                env_block2 = np.einsum('iorq,wiIrRy->wqIoRy',qu_idx,env_block2)
                env_block2 = np.einsum('LIORd,wqIoRy->wqLoOdy',s2d,env_block2)
                env_block2 = np.einsum('wqLoOdy,zoOy->wqLdz',env_block2,bO[1])
                env_out = np.einsum('wlLz,wqLdz->qld',env_block1,env_block2)

        else:  raise (IndexError)

        if return_all:
            if return_corners:     return env_out, qu_idx, qd_idx, axT_inv, None
            else:                  return env_out, qu_idx, qd_idx, axT_inv
        else:
            if return_corners:     return env_out, None
            else:                  return env_out
        

    elif (L1,L2) == (2,1):   # vertical trotter step
    
        if x_idx == (0,0):
            ## update site 2
            s2u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1])
            s2d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1])
    
            # env_block2 = np.einsum('wlLx,liord->wLiorxd',bL[1],s2u)
            # env_block2 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block2,s2d)     # left boundary
            # env_block2 = np.einsum('xoOy,wiIoOrRx->wiIrRy',bO[0],env_block2)     # outer boundary
            # env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])         # right boundary
            env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
            env_block2 = np.einsum('wlLoOy,liord->wLiOrdy',env_block2,s2u)
            env_block2 = np.einsum('wLiOrdy,LIORd->wiIrRy',env_block2,s2d)
            env_block2 = np.einsum('wiIrRy,zrRy->wiIz',env_block2,bR[1])

            ## site 1 boundary
            # assume iso_leg = O
            if idx_key == 'xx':
                env_block1 = np.einsum('wlLx,wiIy->xlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('xlLiIy,lirq->xLIqry',env_block1,qd_idx)
                env_block1 = np.einsum('xLIqry,LIRQ->xqQrRy',env_block1,qu_idx)
                env_block1 = np.einsum('xqQrRy,yrRz->xqQz',env_block1,bR[0])
                env_out = np.einsum('xqQz,xoOz->qQoO',env_block1,env_block2)
            elif idx_key == 'xo':
                s1d = PEPX_GL.get_site(sub_peps_d,(0,0))
                env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
                env_block1 = np.einsum('wlLiIy,lirq->wLIqry',env_block1,qu_idx)
                env_block1 = np.einsum('wLIqry,LIORd->wqOrRdy',env_block1,s1d)
                env_block1 = np.einsum('wqOrRdy,yrRz->wqOdz',env_block1,bR[0])
                env_out = np.einsum('wqOdz,woOz->qod',env_block1,env_block2)


        elif x_idx == (1,0):
            ## update site 1
            s1u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2])
            s1d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2])
        
            # env_block1 = np.einsum('xlLw,liord->wLiorxd',bL[0],s1u)
            # env_block1 = np.einsum('wLiorxd,LIORd->wiIoOrRx',env_block1,s1d)     # left boundary
            # env_block1 = np.einsum('xiIy,wiIoOrRx->woOrRy',bI[0],env_block1)     # inner boundary
            # env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])         # right boundary
            env_block1 = np.einsum('xlLw,xiIy->wlLiIy',bL[0],bI[0])
            env_block1 = np.einsum('wlLiIy,liord->wLIordy',env_block1,s1u)   # assume sum(shape(s1u)) < sum(sh(s2u))
            env_block1 = np.einsum('wLIordy,LIORd->woOrRy',env_block1,s1d)
            env_block1 = np.einsum('woOrRy,yrRz->woOz',env_block1,bR[0])

            ## site 2 boundary
            # assume iso_leg = I
    
            if idx_key == 'xx':
                env_block2 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                env_block2 = np.einsum('xlLoOz,lorq->xLqOrz',env_block2,qu_idx)
                env_block2 = np.einsum('xLqOrz,LORQ->xqQrRz',env_block2,qd_idx)
                env_block2 = np.einsum('xqQrRz,yrRz->xqQy',env_block2,bR[1])
                env_out = np.einsum('xiIz,xqQz->qQiI',env_block1,env_block2)
            elif idx_key == 'xo':
                s2d = PEPX_GL.get_site(sub_peps_d,(1,0))
                env_block2 = np.einsum('wlLx,xoOy->wlLoOy',bL[1],bO[0])
                env_block2 = np.einsum('wlLoOy,lorq->wLqOry',env_block2,qu_idx)
                env_block2 = np.einsum('wLqOry,LIORd->wqIrRdy',env_block2,s2d)
                env_block2 = np.einsum('wqIrRdy,zrRy->wqIdz',env_block2,bR[1])
                env_out = np.einsum('wiIz,wqIdz->qid',env_block1,env_block2)

        else:  raise (IndexError)

        if return_all:
            if return_corners:     return env_out, qu_idx, qd_idx, axT_inv, None
            else:                  return env_out, qu_idx, qd_idx, axT_inv
        else:
            if return_corners:     return env_out, None
            else:                  return env_out
   

    elif (L1,L2) == (2,2):  # LR/square trotter step

        # def get_bL10(bL1,tens10_u,tens10_d,bO0):
        #     if np.sum(tens10_u.shape) > np.sum(tens10_d.shape):
        #         bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        #         bL10 = np.einsum('wlLoOy,LIORd->wlIoRyd',bL10,tens10_d)
        #         bL10 = np.einsum('wlIoRyd,liord->wiIrRy',bL10,tens10_u)
        #     else:
        #         bL10 = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
        #         bL10 = np.einsum('wlLoOy,liord->wLiOryd',bL10,tens10_u)
        #         bL10 = np.einsum('wLiOryd,LIORd->wiIrRy',bL10,tens10_d)
        #     return bL10
 
        # def get_bL11(bO1,tens11_u,tens11_d,bR1):
        #     if np.sum(tens11_u.shape) > np.sum(tens11_d.shape):
        #         bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        #         bL11 = np.einsum('xoOrRz,LIORd->xLIorzd',bL11,tens11_d)
        #         bL11 = np.einsum('xLIorzd,liord->xlLiIz',bL11,tens11_u)
        #     else:
        #         bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
        #         bL11 = np.einsum('xoOrRz,liord->xliORzd',bL11,tens11_u)
        #         bL11 = np.einsum('xliORzd,LIORd->xlLiIz',bL11,tens11_d)
        #     return bL11

        # def get_bL01(bI1,tens01_u,tens01_d,bR0):
        #     if np.sum(tens01_u.shape) > np.sum(tens01_d.shape):
        #         bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        #         bL01 = np.einsum('xiIrRz,LIORd->xLiOrzd',bL01,tens01_d)
        #         bL01 = np.einsum('xLiOrzd,liord->xlLoOz',bL01,tens01_u)
        #     else:
        #         bL01 = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
        #         bL01 = np.einsum('xiIrRz,liord->xlIoRzd',bL01,tens01_u)
        #         bL01 = np.einsum('xlIoRzd,LIORd->xlLoOz',bL01,tens01_d)
        #     return bL01

        # def get_bL00(bL0,tens00_u,tens00_d,bI0):
        #     if np.sum(tens00_u.shape) > np.sum(tens00_d.shape):
        #         bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        #         bL00 = np.einsum('xlLiIz,LIORd->xliORzd',bL00,tens00_d)
        #         bL00 = np.einsum('xliORzd,liord->xoOrRz',bL00,tens00_u)
        #     else:
        #         bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
        #         bL00 = np.einsum('xlLiIz,liord->xLIorzd',bL00,tens00_u)
        #         bL00 = np.einsum('xLIorzd,LIORd->xoOrRz',bL00,tens00_d)
        #     return bL00

        def get_bL10(bL10x=None,no_lam=[]):
            # bL10 = np.einsum('wlLx,liord,LIORd->wiIoOrRx',bL[1],sub_peps_u[1,0],sub_peps_d[1,0])

            bL1, bO0 = bL[1], bO[0]
            tens10_u = PEPX_GL.get_site(sub_peps_u,(1,0),no_lam=[1,3])
            tens10_d = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[1,3])

            if bL10x is None:
                if np.sum(tens10_u.shape) > np.sum(tens10_d.shape):
                    # bL10 = np.einsum('wlLx,LIORd->wL=lIORdx',bL1,tens10_d)
                    # bL10 = np.einsum('wlIORdx,liord->wiIoOrRx',bL10,tens10_u)
                    # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                    bL10x = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                    bL10x = np.einsum('wlLoOy,LIORd->wlIoRyd',bL10x,tens10_d)
                    bL10x = np.einsum('wlIoRyd,liord->wiIrRy',bL10x,tens10_u)
                else:
                    # bL10 = np.einsum('wlLx,liord->wLiordx',bL1,tens10_u)
                    # bL10 = np.einsum('wLiordx,LIORd->wiIoOrRx',bL10,tens10_d)
                    # bL10 = np.einsum('wiIoOrRx,xoOy->wiIrRy',bL10,bO0)
                    bL10x = np.einsum('wlLx,xoOy->wlLoOy',bL1,bO0)
                    bL10x = np.einsum('wlLoOy,liord->wLiOryd',bL10x,tens10_u)
                    bL10x = np.einsum('wLiOryd,LIORd->wiIrRy',bL10x,tens10_d)

            bL10 = bL10x
            if 1 not in no_lam:
                bL10 = tf.dot_diag(bL10,sub_peps_u.lambdas[1,0,1],1)
                bL10 = tf.dot_diag(bL10,sub_peps_d.lambdas[1,0,1],2)
            if 3 not in no_lam:
                bL10 = tf.dot_diag(bL10,sub_peps_u.lambdas[1,0,3],3)
                bL10 = tf.dot_diag(bL10,sub_peps_d.lambdas[1,0,3],4)

            return bL10, bL10x
 
        def get_bL11(bL11x=None,no_lam=[]):
            # bL11 = np.einsum('xoOy,liord,LIORd->xlLiIrRy',bO[1],sub_peps_u[1,1],sub_peps_d[1,1])

            bO1, bR1 = bO[1], bR[1]
            tens11_u = PEPX_GL.get_site(sub_peps_u,(1,1),no_lam=[0,1])
            tens11_d = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[0,1])

            if bL11x is None:
                if np.sum(tens11_u.shape) > np.sum(tens11_d.shape):
                    # bL11 = np.einsum('xoOy,LIORd->xLIoRyd',bO1,tens11_d)
                    # bL11 = np.einsum('xLIoRyd,liord->xlLiIrRy',bL11,tens11_u)
                    # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                    bL11x = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                    bL11x = np.einsum('xoOrRz,LIORd->xLIorzd',bL11x,tens11_d)
                    bL11x = np.einsum('xLIorzd,liord->xlLiIz',bL11x,tens11_u)
                else:
                    # bL11 = np.einsum('xoOy,liord->xliOryd',bO1,tens11_u)
                    # bL11 = np.einsum('xliOryd,LIORd->xlLiIrRy',bL11,tens11_d)
                    # bL11 = np.einsum('xlLiIrRy,zrRy->xlLiIz',bL11,bR1)
                    bL11x = np.einsum('xoOy,zrRy->xoOrRz',bO1,bR1)
                    bL11x = np.einsum('xoOrRz,liord->xliORzd',bL11x,tens11_u)
                    bL11x = np.einsum('xliORzd,LIORd->xlLiIz',bL11x,tens11_d)

            bL11 = bL11x
            if 0 not in no_lam:
                bL11 = tf.dot_diag(bL11,sub_peps_u.lambdas[1,1,0],1)
                bL11 = tf.dot_diag(bL11,sub_peps_d.lambdas[1,1,0],2)
            if 1 not in no_lam:
                bL11 = tf.dot_diag(bL11,sub_peps_u.lambdas[1,1,1],3)
                bL11 = tf.dot_diag(bL11,sub_peps_d.lambdas[1,1,1],4)

            return bL11, bL11x

        def get_bL01(bL01x=None,no_lam=[]):
            # bL01 = np.einsum('xiIy,liord,LIORd->xlLoOrRy',bI[1],sub_peps_u[0,1],sub_peps_d[0,1])

            bI1, bR0 = bI[1], bR[0]
            tens01_u = PEPX_GL.get_site(sub_peps_u,(0,1),no_lam=[0,2])
            tens01_d = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[0,2])

            if bL01x is None:
                if np.sum(tens01_u.shape) > np.sum(tens01_d.shape):
                    # bL01 = np.einsum('xiIy,LIORd->xLiORyd',bI1,tens01_d)
                    # bL01 = np.einsum('xLiORyd,liord->xlLoOrRy',bL01,tens01_u)
                    # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                    bL01x = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                    bL01x = np.einsum('xiIrRz,LIORd->xLiOrzd',bL01x,tens01_d)
                    bL01x = np.einsum('xLiOrzd,liord->xlLoOz',bL01x,tens01_u)
                else:
                    # bL01 = np.einsum('xiIy,liord->xlIoryd',bI1,tens01_u)
                    # bL01 = np.einsum('xlIoryd,LIORd->xlLoOrRy',bL01,tens01_d)
                    # bL01 = np.einsum('xlLoOrRy,yrRz->xlLoOz',bL01,bR0)
                    bL01x = np.einsum('xiIy,yrRz->xiIrRz',bI1,bR0)
                    bL01x = np.einsum('xiIrRz,liord->xlIoRzd',bL01x,tens01_u)
                    bL01x = np.einsum('xlIoRzd,LIORd->xlLoOz',bL01x,tens01_d)

            bL01 = bL01x
            if 0 not in no_lam:
                bL01 = tf.dot_diag(bL01,sub_peps_u.lambdas[0,1,0],1)
                bL01 = tf.dot_diag(bL01,sub_peps_d.lambdas[0,1,0],2)
            if 2 not in no_lam:
                bL01 = tf.dot_diag(bL01,sub_peps_u.lambdas[0,1,2],3)
                bL01 = tf.dot_diag(bL01,sub_peps_d.lambdas[0,1,2],4)

            return bL01, bL01x

        def get_bL00(bL00x=None,no_lam=[]):
            # bL00 = np.einsum('ylLx,liord,LIORd->xiIoOrRy',bL[0],sub_peps_u[0,0],sub_peps_d[0,0])

            bI0, bL0 = bI[0], bL[0]
            tens00_u = PEPX_GL.get_site(sub_peps_u,(0,0),no_lam=[2,3])
            tens00_d = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[2,3])

            if bL00x is None:
                if np.sum(tens00_u.shape) > np.sum(tens00_d.shape):
                    # bL00 = np.einsum('ylLx,LIORd->xlIORyd',bL0,tens00_d)
                    # bL00 = np.einsum('xlIORyd,liord->xiIoOrRy',bL00,tens00_u)
                    # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                    bL00x = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                    bL00x = np.einsum('xlLiIz,LIORd->xliORzd',bL00x,tens00_d)
                    bL00x = np.einsum('xliORzd,liord->xoOrRz',bL00x,tens00_u)
                else:
                    # bL00 = np.einsum('ylLx,liord->xLioryd',bL0,tens00_u)
                    # bL00 = np.einsum('xLioryd,LIORd->xiIoOrRy',bL00,tens00_d)
                    # bL00 = np.einsum('xiIoOrRy,yiIz->xoOrRz',bL00,bI0)
                    bL00x = np.einsum('ylLx,yiIz->xlLiIz',bL0,bI0)
                    bL00x = np.einsum('xlLiIz,liord->xLIorzd',bL00x,tens00_u)
                    bL00x = np.einsum('xLIorzd,LIORd->xoOrRz',bL00x,tens00_d)

            bL00 = bL00x
            if 2 not in no_lam:
                bL00 = tf.dot_diag(bL00,sub_peps_u.lambdas[0,0,2],1)
                bL00 = tf.dot_diag(bL00,sub_peps_d.lambdas[0,0,2],2)
            if 3 not in no_lam:
                bL00 = tf.dot_diag(bL00,sub_peps_u.lambdas[0,0,3],3)
                bL00 = tf.dot_diag(bL00,sub_peps_d.lambdas[0,0,3],4)

            return bL00, bL00x


        if old_corners is not None:           bL00x, bL01x, bL10x, bL11x = old_corners
        else:                                 bL00x, bL01x, bL10x, bL11x = [None,None,None,None]

        ### order contractions assuming 'rol' mpo connectivity ###
        ### so that untruncated bond dims 00 ~ 10 < 01 ~ 11 for 4-body operator
        ###                               10 < 00 ~ 11 < 01 for 3-body operator
        if x_idx == (0,0):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[1,3])
            bL11,bL11x = get_bL11(bL11x,no_lam=[])
            bL01,bL01x = get_bL01(bL01x,no_lam=[0,2])

            # bL10_,bL10x_ = get_bL10(None,no_lam=[1,3])
            # bL11_,bL11x_ = get_bL11(None,no_lam=[])
            # bL01_,bL01x_ = get_bL01(None,no_lam=[0,2])

            # err = [np.linalg.norm(bL10-bL10_),np.linalg.norm(bL11-bL11_),np.linalg.norm(bL01-bL01_)]
            # if np.any(np.array(err)>1.0e-10):                print '(0,0)',err, [(c is None) for c in old_corners]

            ## 10 -> 11 -> 01
            bLs = np.einsum('wiIrRx,xrRoOy->wiIoOy',bL10,bL11)
            bLs = np.einsum('wiIoOy,zlLoOy->wiIlLz',bLs, bL01)

            ## 00 corner 
            if idx_key == 'xx':    ## metric
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                if   iso_leg in ['r','R',3]:
                    bL00 = np.einsum('xlLiIz,lioq->xLIoqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIoqz,LIOQ->xoOqQz',bL00,qd_idx)
                    env_out = np.einsum('woOqQz,woOrRz->qQrR',bL00,bLs)
                elif iso_leg in ['o','O',2]:
                    bL00 = np.einsum('xlLiIz,lirq->xLIqrz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqrz,LIRQ->xqQrRz',bL00,qd_idx)
                    env_out = np.einsum('wqQrRz,woOrRz->qQoO',bL00,bLs)
                elif iso_leg in ['or','ro','OR','RO']:
                    bL00 = np.einsum('xlLiIz,liq->xLIqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqz,LIQ->xqQz',bL00,qd_idx)
                    env_out = np.einsum('wqQz,woOrRz->qQoOrR',bL00,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':
                sd00 = PEPX_GL.get_site(sub_peps_d,(0,0),no_lam=[])
                bL00 = np.einsum('ylLx,yiIz->xlLiIz',bL[0],bI[0])
                if   iso_leg in ['r','R',3]:
                    bL00 = np.einsum('xlLiIz,lioq->xLIoqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIoqz,LIORd->xoOqRdz',bL00,sd00)
                    env_out = np.einsum('xoOqRdz,xoOrRz->qrd',bL00,bLs)
                elif iso_leg in ['o','O',2]:
                    bL00 = np.einsum('xlLiIz,lirq->xLIqrz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqrz,LIORd->xqOrRdz',bL00,sd00)
                    env_out = np.einsum('xqOrRdz,xoOrRz->qod',bL00,bLs)
                elif iso_leg in ['or','ro','OR','RO']:
                    bL00 = np.einsum('xlLiIz,liq->xLIqz',bL00,qu_idx)
                    bL00 = np.einsum('xLIqz,LIORd->xqORdz',bL00,sd00)
                    env_out = np.einsum('wqORdz,woOrRz->qord',bL00,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [None,bL01x,bL10x,bL11x]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [None,bL01x,bL10x,bL11x]
                else:                 return env_out

        elif x_idx == (0,1):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[])
            bL11,bL11x = get_bL11(bL11x,no_lam=[0,1])
            bL00,bL00x = get_bL00(bL00x,no_lam=[2,3])

            # bL10_,bL10x_ = get_bL10(None,no_lam=[])
            # bL11_,bL11x_ = get_bL11(None,no_lam=[0,1])
            # bL00_,bL00x_ = get_bL00(None,no_lam=[2,3])

            # err = [np.linalg.norm(bL10-bL10_),np.linalg.norm(bL11-bL11_),np.linalg.norm(bL00-bL00_)]
            # if np.any(np.array(err)>1.0e-10):                print '(0,1)', err, [(c is None) for c in old_corners]

            ## 00 -> 10 -> 11
            bLs = np.einsum('xoOrRw,xoOlLy->wrRlLy',bL00,bL10)
            bLs = np.einsum('wrRlLy,ylLiIz->wrRiIz',bLs ,bL11)

            ## 01 corner
            if idx_key == 'xx':    ## metric
                bL01 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                if   iso_leg in ['l','L',0]:
                    bL01 = np.einsum('xiIrRz,iorq->xqIoRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIoRz,IORQ->xqQoOz',bL01,qd_idx)
                    env_out = np.einsum('xqQoOz,xlLoOz->qQlL',bL01,bLs)
                elif iso_leg in ['o','O',2]:
                    bL01 = np.einsum('xiIrRz,lirq->xlIqRz',bL01,qu_idx)
                    bL01 = np.einsum('xlIqRz,LIRQ->xlLqQz',bL01,qd_idx)
                    env_out = np.einsum('xlLqQz,xlLoOz->qQoO',bL01,bLs)
                elif iso_leg in ['lo','LO','ol','OL']:
                    bL01 = np.einsum('xiIrRz,irq->xqIRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIRz,IRQ->xqQz',bL01,qd_idx)
                    env_out = np.einsum('xqQz,xlLoOz->qQlLoO',bL01,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':  ## gradient
                sd01 = PEPX_GL.get_site(sub_peps_d,(0,1),no_lam=[])
                bL01 = np.einsum('wiIx,xrRy->wiIrRy',bI[1],bR[0])
                if   iso_leg in ['l','L',0]:
                    bL01 = np.einsum('xiIrRz,iorq->xqIoRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIoRz,LIORd->xqLoOdz',bL01,sd01)
                    env_out = np.einsum('xqLoOdz,xlLoOz->qld',bL01,bLs)
                elif iso_leg in ['o','O',2]:
                    bL01 = np.einsum('xiIrRz,lirq->xlIqRz',bL01,qu_idx)
                    bL01 = np.einsum('xlIqRz,LIORd->xlLqOdz',bL01,sd01)
                    env_out = np.einsum('xlLqOdz,xlLoOz->qod',bL01,bLs)
                elif iso_leg in ['lo','LO','ol','OL']:
                    bL01 = np.einsum('xiIrRz,irq->xqIRz',bL01,qu_idx)
                    bL01 = np.einsum('xqIRz,LIORd->xqLOdz',bL01,sd01)
                    env_out = np.einsum('xqLOdz,xlLoOz->qlod',bL01,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [bL00x,None,bL10x,bL11x]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [bL00x,None,bL10x,bL11x]
                else:                 return env_out

        elif x_idx == (1,0):

            # update other corners
            bL01,bL01x = get_bL01(bL01x,no_lam=[])
            bL11,bL11x = get_bL11(bL11x,no_lam=[0,1])
            bL00,bL00x = get_bL00(bL00x,no_lam=[2,3])

            # bL01_,bL01x_ = get_bL01(None,no_lam=[])
            # bL11_,bL11x_ = get_bL11(None,no_lam=[0,1])
            # bL00_,bL00x_ = get_bL00(None,no_lam=[2,3])

            # err = [np.linalg.norm(bL00-bL00_),np.linalg.norm(bL11-bL11_),np.linalg.norm(bL01-bL01_)]
            # if np.any(np.array(err)>1.0e-10):                print '(1,0)',err, [(c is None) for c in old_corners]

            ## 00 -> 01 -> 11
            bLs = np.einsum('woOrRx,xrRiIy->woOiIy',bL00,bL01)
            bLs = np.einsum('woOiIy,zlLiIy->woOlLz',bLs, bL11)
 
            ## 10 corner
            if idx_key == 'xx':
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                if   iso_leg in ['i','I',1]:
                    bL10 = np.einsum('xlLoOz,lorq->xLqOrz',bL10,qu_idx)
                    bL10 = np.einsum('xLqOrz,LORQ->xqQrRz',bL10,qd_idx)
                    env_out = np.einsum('xqQrRz,xiIrRz->qQiI',bL10,bLs)
                elif iso_leg in ['r','R',3]:
                    bL10 = np.einsum('xlLoOz,lioq->xLiOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLiOqz,LIOQ->xiIqQz',bL10,qd_idx)
                    env_out = np.einsum('xiIqQz,xiIrRz->qQrR',bL10,bLs)
                elif iso_leg in ['ri','ir','RI','IR']:
                    bL10 = np.einsum('xlLoOz,loq->xLOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLOqz,LOQ->xqQz',bL10,qd_idx)
                    env_out = np.einsum('xqQz,xiIrRz->qQiIrR',bL10,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':
                sd10 = PEPX_GL.get_site(sub_peps_d,(1,0),no_lam=[])
                bL10 = np.einsum('xlLy,yoOz->xlLoOz',bL[1],bO[0])
                if   iso_leg in ['i','I',1]:
                    bL10 = np.einsum('xlLoOz,lorq->xLqOrz',bL10,qu_idx)
                    bL10 = np.einsum('xLqOrz,LIORd->xqIrRdz',bL10,sd10)
                    env_out = np.einsum('xqIrRdz,xiIrRz->qid',bL10,bLs)
                elif iso_leg in ['r','R',3]:
                    bL10 = np.einsum('xlLoOz,lioq->xLiOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLiOqz,LIORd->xiIqRdz',bL10,sd10)
                    env_out = np.einsum('xiIqRdz,xiIrRz->qrd',bL10,bLs)
                elif iso_leg in ['ri','ir','RI','IR']:
                    bL10 = np.einsum('xlLoOz,loq->xLOqz',bL10,qu_idx)
                    bL10 = np.einsum('xLOqz,LIORd->xqIRdz',bL10,sd10)
                    env_out = np.einsum('xqIRdz,xiIrRz->qird',bL10,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [bL00x,bL01x,None,bL11x]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [bL00x,bL01x,None,bL11x]
                else:                 return env_out


        elif x_idx == (1,1):

            # update other corners
            bL10,bL10x = get_bL10(bL10x,no_lam=[1,3])
            bL00,bL00x = get_bL00(bL00x,no_lam=[])
            bL01,bL01x = get_bL01(bL01x,no_lam=[0,2])

            ## 00 -> 10 -> 01
            bLs = np.einsum('xiIrRw,xiIlLy->wrRlLy',bL10,bL00)
            bLs = np.einsum('wrRlLy,ylLoOz->wrRoOz',bLs, bL01)

            ## 11 corner
            if idx_key == 'xx':
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                if   iso_leg in ['l','L',0]:
                    bL11 = np.einsum('xoOrRz,iorq->xqiORz',bL11,qu_idx)
                    bL11 = np.einsum('xqiORz,IORQ->xqQiIz',bL11,qd_idx)
                    env_out = np.einsum('xqQiIz,xlLiIz->qQlL',bL11,bLs)
                elif iso_leg in ['i','I',1]:
                    bL11 = np.einsum('xoOrRz,lorq->xlqORz',bL11,qu_idx)
                    bL11 = np.einsum('xlqORz,LORQ->xlLqQz',bL11,qd_idx)
                    env_out = np.einsum('xlLqQz,xlLiIz->qQiI',bL11,bLs)
                elif iso_leg in ['li','il','LI','IL']:
                    bL11 = np.einsum('xoOrRz,orq->xqORz',bL11,qu_idx)
                    bL11 = np.einsum('xqORz,ORQ->xqQz',bL11,qd_idx)
                    env_out = np.einsum('xqQz,xlLiIz->qQlLiI',bL11,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'
            elif idx_key == 'xo':
                sd11 = PEPX_GL.get_site(sub_peps_d,(1,1),no_lam=[])
                bL11 = np.einsum('xoOy,zrRy->xoOrRz',bO[1],bR[1])
                if   iso_leg in ['l','L',0]:
                    bL11 = np.einsum('xoOrRz,iorq->xqiORz',bL11,qu_idx)
                    bL11 = np.einsum('xqiORz,LIORd->xqLiIdz',bL11,sd11)
                    env_out = np.einsum('xqLiIdz,xlLiIz->qld',bL11,bLs)
                elif iso_leg in ['i','I',1]:
                    bL11 = np.einsum('xoOrRz,lorq->xlqORz',bL11,qu_idx)
                    bL11 = np.einsum('xlqORz,LIORd->xlLqIdz',bL11,sd11)
                    env_out = np.einsum('xlLqIdz,xlLiIz->qid',bL11,bLs)
                elif iso_leg in ['li','il','LI','IL']:
                    bL11 = np.einsum('xoOrRz,orq->xqORz',bL11,qu_idx)
                    bL11 = np.einsum('xqORz,LIORd->xqLIdz',bL11,sd11)
                    env_out = np.einsum('xqLIdz,xlLiIz->qlid',bL11,bLs)
                else:
                    raise(IndexError),'need valid iso_leg'

            if return_all:
                if return_corners:    return env_out, qu_idx, qd_idx, axT_inv, [bL00x,bL01x,bL10x,None]
                else:                 return env_out, qu_idx, qd_idx, axT_inv
            else:
                if return_corners:    return env_out, [bL00x,bL01x,bL10x,None]
                else:                 return env_out

        else:  raise (IndexError)

    else:
        raise (NotImplementedError)
     


