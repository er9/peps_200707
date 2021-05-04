import numpy as np
import time
import tens_fcts as tf

import MPX as MPX
import PEPX
import PEPX_GL
import PEPS_env as ENV
import PEPS_GL_env_nolam as ENV_GL
import TLDM_GL
import TLDM_GL_env_nolam as ENV_DM





#####################################################################
###        alternating least square methods                       ###
#####################################################################

def optimize_disentangler(g_env):
    ''' rerturn optimized disentangler g
        two-site entangler, aAbB where a,b -> bra ancilla; A,B -> ket ancilla 

        g_env = ABab (ket ancilla) x (bra ancilla)
    '''

    g_shape = g_env.shape
    sqdim = int(np.sqrt(np.prod(g_shape)))

    g_block = g_env.reshape(sqdim,sqdim)
    u,s,vt = np.linalg.svd(g_block)
    g = np.tensordot( np.conj(u), np.conj(vt), axes=(1,0) ).reshape(g_shape)   #(AB)x(ab) -> ABab

    ## check:
    val = np.tensordot(g_env,g,axes=(range(g_env.ndim),range(g.ndim)))
    # print 'opt disentangler', val, np.sum(s)
    if np.abs(val - np.sum(s)) > 1.0e-12:
        print 'g not optimal?'

    return g


def pos_pinv(sq_mat, rcond=1.0e-15, ensure_pos=False):

    if ensure_pos:
    
        isH = np.allclose(sq_mat,np.conj(sq_mat.T))

        if isH:            return np.linalg.pinv(sq_mat,rcond=rcond) # ,hermitian=True) ## need np 1.17 (i'm 1.16.5)
        else:    # find approx positive matrix

            print 'ensuring pos'

            # ### Reza's method:   sqrt(M^* M)
            # u,s,vt = np.linalg.svd(sq_mat)
            # # sqrt_M2 = np.dot(np.conj(vt.T), tf.dMult('DM',s,vt))  
            # 
            # s_ = s[abs(s) > rcond*s[0]]
            # M  = len(s_)
            # pseudo_inv = np.dot(np.conj(vt[:M,:].T), tf.dMult('DM',1./s_,vt[:M,:]))


            ### Lubasch's method (M^+ + M)/2
            sym_mat = (sq_mat + np.conj(sq_mat.T))/2
            pseudo_inv = np.linalg.pinv(sym_mat,rcond=rcond)

            return pseudo_inv

    else:
        return np.linalg.pinv(sq_mat,rcond=rcond)
        


def optimize_site(site_metric,site_grad,rcond=1.0e-8,ensure_pos=False):
    ''' return one optimized site
       (site_metric -- reshape into sq matrix split along bonds for the two missing sites)
       (vector
    '''

    site_shape = site_metric.shape
    sqdim = int(np.sqrt(np.prod(site_metric.shape)))
    M_block = site_metric.transpose(0,2,4,6,1,3,5,7).reshape(sqdim,sqdim)   # lLiIoOrR -> (lior)x(LIOR)
    G_vec = site_grad.reshape(sqdim,-1)    # (lior)x(da)

    # print 'is hermitian?', np.allclose(M_block,np.conj(M_block.T))

    # M_inv = np.linalg.pinv(M_block, rcond=1.0e-8)
    M_inv = pos_pinv(M_block, rcond=rcond, ensure_pos=ensure_pos)
    site_ = np.matmul(M_inv,G_vec)

    # if np.linalg.norm(site_) > 5.0:
    #     print 'optimize site large norm', np.linalg.norm(site_), np.linalg.norm(G_vec),
    #     print np.linalg.norm(M_block), np.linalg.norm(M_inv)
    #     # exit()

    return site_.reshape(site_grad.shape)


def red_optimize_site(site_metric,site_grad):
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

    M_inv = np.linalg.pinv(M_block, rcond=1.0e-8)
    site_ = np.matmul(M_inv,G_vec)

    return site_.reshape(site_grad.shape)



####################################################
#####               update methods           #######
####################################################

def simple_update(gamma_list, lambda_list, block1, DMAX=10, direction=0, normalize=True):
    ''' algorithm:  contract all sites into block and then do svd compression
                    not the most efficient, but maybe numerically more stable?
        pepo_part:  list of pepo sites that we're operating on
        block1:     nparray mpo-like block (1 x ... x 1) applied to sites on 'i' vertical bond
    '''    

    if isinstance(block1,MPX.MPX):   # this needs to go first bc MPX is also instance of np.ndarray
        new_gams, new_lams, errs = PEPX_GL.mpo_update(gamma_list, lambda_list, block1, DMAX=DMAX,
                                                      direction=direction,normalize=normalize) 

    elif isinstance(block1, np.ndarray):
        new_gams, new_lams, errs = PEPX_GL.block_update(gamma_list, lambda_list, block1, DMAX=DMAX,
                                                        direction=direction,normalize=normalize)
    else:
        raise(TypeError), 'block1 should be np.ndarray (block) or MPX.MPX (mpo)'

    return new_gams, new_lams, errs


# @profile
def alsq_block_update(sub_tldm_gl,connect_list,inds_list,env_list,trotter_block,DMAX=10,XMAX=100,site_idx=None,
                      init_guess=None, ensure_pos=False, normalize=False, regularize=True, build_env=False,
                      qr_reduce=False):
    ''' optimize each tens in peps_list, given environment
        env_list:  [left, inner, outer, right] boundary mpo's
        inds_list: [xs,ys]
        site_idx = list of sites to do alsq over. if None, by default all sites are optimized over
    '''

    sub_pepx_gl = PEPX_GL.PEPX_GL(sub_tldm_gl)

    L1,L2 = sub_pepx_gl.shape
    xs,ys = inds_list
 
    ind0 = (inds_list[0][0],inds_list[1][0])
    gam_list, lam_list, axT_invs = PEPX_GL.get_sites(sub_pepx_gl,ind0,connect_list)

    # apply trotter and then svd
    gam_dt, lam_dt, errs = simple_update(gam_list,lam_list,trotter_block,DMAX=-1,direction=0)
    pepx_dt_gl = PEPX_GL.set_sites(sub_pepx_gl,ind0,connect_list,gam_dt,lam_dt,axT_invs) 


    if normalize:
        dt_norm = ENV_DM.embed_sites_norm(pepx_dt_gl, env_list)
        pepx_dt_gl = PEPX_GL.mul(1./dt_norm, pepx_dt_gl)


    if isinstance(init_guess,np.ndarray): 
         # init guess -- increase bond dimension using svd init guess
         leg_ind = PEPX.leg2ind(connect_list[0])
         if sub_tldm_gl[0,0].shape[leg_ind] < DMAX:
             pepx_opt_gl = PEPX_GL.compress(pepx_dt_gl,DMAX)
         else:
             pepx_opt_gl = init_guess

    elif init_guess in ['rand','random']:
         leg_ind_o = [PEPX_GL.leg2ind(c_leg) for c_leg in connect_list]+[None]
         leg_ind_i = [None]+[PEPX_GL.leg2ind(PEPX_GL.opposite_leg(c_leg)) for c_leg in connect_list]
         shape_array = np.empty((L1,L2),dtype=tuple)
         for i in range(len(gam_list)):
             ind = (xs[i],ys[i])
             rand_shape = list(peps_list[i].shape)
             if leg_ind_o[i] is not None:     rand_shape[leg_ind_o[i]] = DMAX
             if leg_ind_i[i] is not None:     rand_shape[leg_ind_i[i]] = DMAX
             shape_array[ind] = tuple(rand_shape)

         raise(NotImplementedError), 'random init state for alsq not implemented'

    else:     # init_guess is None or init_guess in ['su','SU','svd']:
         pepx_opt_gl = PEPX_GL.compress(pepx_dt_gl,DMAX)

    old_pepx_opt_gl = pepx_opt_gl



    # if site_idx is None:        site_idx = [m for m in np.ndindex(sub_peps_gl.shape)]
    # else:                       site_idx = [m for m in site_idx]

    # if site_idx is None:         site_idx = [(xs[i],ys[i]) for i in range(len(xs))]
    # else:                        site_idx = [m for m in site_idx]

    if site_idx is None:
        if (L1,L2) == (2,2):     site_idx = [(0,0),(0,1),(1,1),(1,0)]
        else:                    site_idx = [(xs[i],ys[i]) for i in range(len(xs))]
    else:                        site_idx = [m for m in site_idx]

    # if site_idx is None:
    #     if (L1,L2) == (2,2):     site_idx = [(0,0),(0,1),(1,1),(1,0)]
    #     else:                    site_idx = [m for m in np.ndindex(sub_peps.shape)]
    # else:                        site_idx = [m for m in site_idx]


    # print 'site idx', site_idx

    not_converged = False
    fidelities = []
    prev_alsq_diff = 100
    prev_fidelity  = 0
    it = 0

    # opt_norm = ENV.embed_sites_norm_env(peps_opt, norm_env)

    # print 'init guess', [np.linalg.norm(m) for idx,m in np.ndenumerate(peps_opt)]

    # if qr_reduce:
    temp_xs = [m[0] for m in site_idx]
    temp_ys = [m[1] for m in site_idx]
    loop_conn   = PEPX.get_inds_conn([temp_xs,temp_ys],wrap_inds=True)
    loop_conn_r = PEPX.get_inds_conn([temp_xs[::-1],temp_ys[::-1]],wrap_inds=True)
    # print site_idx, temp_xs, temp_ys, loop_conn, loop_conn_r

    site_list = site_idx
    site_conn = loop_conn
    n_opt = len(site_list)


    while (not_converged or it < 3) and it < 500:

        idx_ind = 0

        if qr_reduce and len(site_idx) > 2:
            if it%2==0:  
                site_list = site_idx
                site_conn = loop_conn
            else:
                site_list = site_idx[::-1]
                site_conn = loop_conn_r

        # ## find disentangler
        env_g = ENV_DM.embed_sites_oog(np.conj(pepx_opt_gl), pepx_dt_gl, env_list, site_list[0], site_conn[0])
        opt_g = optimize_disentangler(env_g)
        sqdim = int(np.sqrt(np.prod(opt_g.shape)))
        # opt_g = np.eye(sqdim).reshape(opt_g.shape)

        # temp_g = opt_g.transpose(0,2,1,3)
        # temp_g = temp_g.reshape((1,)+temp_g.shape+(1,))
        # g_mpo = tf.decompose_block( temp_g, 2, 0, -1, svd_str='iii,...' )

        # ## update pepx_dt_gl with disentangler
        # gs, ls, axTs = PEPX_GL.get_sites(pepx_dt_gl,(0,0),site_conn[0])
        # gs_, ls_, errs = PEPX_GL.mpo_update(gs,ls,None,mpo2=g_mpo,DMAX=DMAX)
        # pepx_opt_gl = PEPX_GL.set_sites(pepx_dt_gl,(0,0),site_conn[0],gs_,ls_,axTs)
        # # print [m.shape for xx,m in np.ndenumerate(pepx_dt_gl)]
        # # pepx_opt_gl = PEPX_GL.compress(pepx_dt_gl,DMAX)

        for idx in site_list:

            iso_leg = site_conn[idx_ind]

            # ## find disentangler
            # # print 'opt',[ m.shape for xx,m in np.ndenumerate(pepx_opt_gl) ]
            # # print 'dt', [ m.shape for xx,m in np.ndenumerate(pepx_dt_gl) ]

            # env_g = ENV_DM.embed_sites_oog(np.conj(pepx_opt_gl), pepx_dt_gl, env_list, idx, iso_leg)
            # opt_g = optimize_disentangler(env_g)
            # # opt_g = np.eye(4).reshape(2,2,2,2)   # ABab



            if not qr_reduce:

                env_x  = ENV_DM.embed_sites_xog(np.conj(pepx_opt_gl),pepx_dt_gl,env_list,idx, opt_g)
                # env_x  = ENV_DM.embed_sites_xo(np.conj(pepx_opt_gl),pepx_dt_gl,env_list,idx)
                env_xx = ENV_DM.embed_sites_xx(np.conj(pepx_opt_gl),pepx_opt_gl,env_list,idx)
                site_opt = optimize_site(env_xx,env_x,ensure_pos=ensure_pos)   # includes lams
                # print 'site opt', site_opt.shape

                # tempx = ENV_DM.embed_sites_xo(np.conj(pepx_opt_gl),pepx_dt_gl,env_list,idx)
                # diff = np.linalg.norm(tempx-env_x)
                # print 'grad diff', idx, iso_leg, diff


            else:
                env_xx,qu_idx,qd_idx,axT_inv_5 = ENV_DM.red_embed_sites_xx(np.conj(pepx_opt_gl),pepx_opt_gl,
                                                                           env_list,idx,iso_leg)
                # axT_inv_5 is for flattened pepx_opt_gl. q's don't change but axT_inv does

                axT, axT_inv = PEPX.get_io_axT('',iso_leg,6,d_end=True)
                env_x = ENV_DM.red_embed_sites_xog(np.conj(pepx_opt_gl),pepx_dt_gl ,env_list,idx,opt_g,iso_leg,
                                                   qu_idx,qd_idx)

                r_opt = red_optimize_site(env_xx,env_x)   # includes lams

                # print axT_inv, iso_leg, pepx_opt_gl[idx].shape, qd_idx.shape, r_opt.shape

                site_opt = PEPX.QR_contract(qd_idx,r_opt,axT_inv,d_end=True)


            ## only change gamma (keep lambda from dt svd)
            # pepx_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,pepx_opt_gl.lambdas[idx])

            ## do svd on site_opt + next site to keep it close to GL form
            pepx_opt_gl[idx] = PEPX_GL.remove_lam_from_site(site_opt,pepx_opt_gl.lambdas[idx])
            gs, ls, axTs = PEPX_GL.get_sites(pepx_opt_gl,idx,iso_leg)
            gs_, ls_ = PEPX_GL.canonicalize_GL_list(gs,ls)
            pepx_opt_gl = PEPX_GL.set_sites(pepx_opt_gl,idx,iso_leg,gs_,ls_,axTs)
            
            idx_ind += 1


            if np.any( [np.linalg.norm(pepx_opt_gl.lambdas[xx]) > 10 for xx in np.ndindex(L1,L2,4)] ):
                print 'WARNING: opt gamma large lam',idx,[np.linalg.norm(m) for x,m in np.ndenumerate(pepx_opt_gl)]
                # print 'WARNING: opt gamma large norm',idx,[np.linalg.norm(m) for x,m in np.ndenumerate(pepx_opt_gl)]
                # print 'env norms', np.linalg.norm(env_x), np.linalg.norm(env_xx), np.linalg.norm(norm_env)

        ############

        if regularize:
            pepx_opt_gl = PEPX_GL.regularize(pepx_opt_gl)
            # print 'regularize', [np.linalg.norm(m) for m in gam_opt], lam_opt

        if build_env:
            raise(NotImplementedError),'did not implement build_env = True'
            opt_norm = ENV_GL.embed_sites_norm_env(pepx_opt_gl, norm_env)
            fidelity = ENV_GL.embed_sites_ovlp_env(np.conj(pepx_dt_gl),pepx_opt_gl,norm_env)/ opt_norm
        else:
            opt_norm = ENV_DM.embed_sites_norm(pepx_opt_gl, env_list)
	    # fidelity = ENV_DM.embed_sites_ovlp(np.conj(pepx_opt_gl),pepx_dt_gl,env_list)/ opt_norm
            fidelity = ENV_DM.embed_sites_ovlp_g(np.conj(pepx_opt_gl), pepx_dt_gl, env_list, opt_g)

            if np.isnan(opt_norm):
                print 'dt norm'
                print ENV_DM.embed_sites_norm(pepx_dt_gl, env_list)
               
                print 'init norm'
                print ENV_DM.embed_sites_norm(sub_pepx_gl, env_list)


                exit()

        # print 'fidelity', it, fidelity, opt_norm

        if np.imag(fidelity) > np.real(fidelity):
            print 'imag fidelity or norm'
            print fidelity, opt_norm
            print 'imag opt', [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(pepx_opt_gl)]
            print 'real opt', [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(pepx_opt_gl)]
            print 'imag dt',  [np.linalg.norm(np.imag(m)) for idx,m in np.ndenumerate(pepx_dt_gl)]
            print 'real dt',  [np.linalg.norm(np.real(m)) for idx,m in np.ndenumerate(pepx_dt_gl)]
            if build_env:  print np.linalg.norm(norm_env)

        try:
            # not_converged = np.any( np.diff(fidelities[-5:]) > 1.0e-5 )
            # not_converged = np.abs(alsq_diff) > 5.0e-5
            # not_converged = (np.abs(prev_alsq_diff) - np.abs(alsq_diff)) > 1.0e-8   # alsq decreases
            not_converged = np.abs(fidelity - prev_fidelity) > 1.0e-10

            if (prev_fidelity > fidelity+1.0e-12):
                pepx_opt_gl = old_pepx_opt_gl
                print 'fidelity went down or alsq diff went up'
                break

            # prev_alsq_diff = alsq_diff
            temp = prev_fidelity
            prev_fidelity     = fidelity
            old_pepx_opt_gl   = pepx_opt_gl

        except(IndexError):   pass

        it += 1

    if not_converged:
        # fidelity = ENV_DM.embed_sites_ovlp_g(np.conj(pepx_opt_gl), pepx_dt_gl, env_list, opt_g)
        # print 'not converged', it, fidelity - prev_fidelity
        print 'not converged', it, fidelity - temp

    if normalize:
        pepx_opt_gl = PEPX_GL.mul(1./opt_norm, pepx_opt_gl)

    return pepx_opt_gl


