import numpy as np
import time

import tens_fcts as tf
import MPX  
import PEPX
import PEPX_GL
import PEPS
import PEPS_GL
import PEPS_env as ENV
import PEPS_GL_env_nolam as ENV_GL
import PEPS_GL_env_nolam_SL as ENV_SL
import Operator_2D as Op
import TimeEvolution as TE

import PEPX_trotterTE as PXTE


# this is for finite PEPS:  so sweep through all of the lattice sites in a dmrg-like style
# calculate entire environment once, and then just make updates with the sweep


'''
general algorithm:  do calcs with initial psi/env
 - apply horizontal bonds + update environment   -> how to update column environments?
 - apply vertical bonds + update environment     -> how to update row environments?
 - apply diagonal bonds 


'''


def get_exp(trotterH,dt,pepx_type='peps'):

    op_keys = trotterH.ops.keys()
    exp_tsteps = {}
    for opk in op_keys:     # trotterOp is in MPO form
         if pepx_type in ['state','peps']:
             # exp_tsteps[opk] = TE.exact_exp(trotterH.ops[opk], dt)

             # print 'mpo type', [m.dtype for m in trotterH.ops[opk]], dt
             temp_exp = TE.exact_exp(trotterH.ops[opk], dt)

             # print 'block exp', temp_exp.dtype
             ns = len(trotterH.conn[opk])+1
             # exp_tsteps[opk] = MPX.MPX(tf.decompose_block(temp_exp,ns,0,-1,'ijk,...'))
             mpo, s_list = tf.decompose_block(temp_exp,ns,0,-1,'ijk,...',return_s=True)
             mpo_ = PEPX.split_singular_vals(mpo,None,s_list)
             exp_tsteps[opk] = MPX.MPX(mpo_)
             # print 'mpo decomposition err', np.linalg.norm((MPX.MPX(mpo_)).getSites()-temp_exp)
             # print 'exp type', [m.dtype for m in mpo]

         elif pepx_type in ['dm','DM','rho','pepo']:
             # exp_tsteps[opk] = TE.exact_expL(trotterH.ops[opk], dt)

             ns = len(trotterH.conn[opk])+1
             temp_exp = TE.exact_expL(trotterH.ops[opk], dt)
             # exp_tsteps[opk] = MPX.MPX(tf.decompose_block(temp_exp,ns,0,-1,'ijk,...'))
             mpo, s_list = tf.decompose_block(temp_exp,ns,0,-1,'ijk,...',return_s=True)
             mpo_ = PEPX.split_singular_vals(mpo,None,s_list)
             exp_tsteps[opk] = MPX.MPX(mpo_)

    return exp_tsteps



### update sites via SU or FU ###

def update_sites_SU(pepx,ind0,op_conn,op,DMAX=10):

    gam, lam, axT_invs = PEPX_GL.get_sites(pepx,ind0,op_conn)
    gam_, lam_, errs = PEPS_GL.simple_update(gam, lam, op, DMAX=DMAX, direction=1, normalize=True)
    # gam_, lam_ = PEPX_GL.regularize_GL_list(gam_,lam_)
    pepx_ = PEPX_GL.set_sites(pepx,ind0,op_conn,gam_,lam_,axT_invs)

    return pepx_


def update_sites_FU(pepx,ind0,op_conn,op,envs_list,DMAX=10,XMAX=100,build_env=False,qr_reduce=False,
                    ensure_pos=False):

    xs,ys = PEPX.get_conn_inds(op_conn,ind0)

    pepx_ = PEPS_GL.alsq_block_update(pepx,op_conn,[xs,ys],envs_list,op,DMAX=DMAX,XMAX=XMAX,site_idx=None,
                                      normalize=False,build_env=build_env,qr_reduce=qr_reduce,
                                      ensure_pos=ensure_pos)#,init_guess=pepx)
    return pepx_
    


### apply layer of trotter steps ###
# @profile
def apply_trotter_layer_SU(pepx, step_list, trotterH, expH=None, DMAX=10):

    pepx_ = pepx.copy()

    for ind, opk in step_list:

        # print ind, opk

        i,j = ind

        NR, NC  = trotterH.ns[opk]
        op_conn = trotterH.conn[opk]
        ind0    = trotterH.ind0[opk]

        if expH is None:      op  = trotterH.ops[opk]
        else:                 op  = expH[opk]

        sub_pepx = pepx_[i:i+NR,j:j+NC]
        sub_pepx = update_sites_SU(sub_pepx,ind0,op_conn,op,DMAX=DMAX)
        
        pepx_[i:i+NR,j:j+NC] = sub_pepx

    return pepx_


# @profile
def apply_trotter_layer_IO(pepx, step_list, trotterH, expH=None, DMAX=10, XMAX=100, build_env=False, 
                           qr_reduce=False, ensure_pos=False):
    ''' perform trotter step using I and O as main boundaries'''

    pepx_ = pepx.copy()

    envIs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'I', 0, XMAX=XMAX)  # list of len ii+1
    envOs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'O', 0, XMAX=XMAX)  # list of len L+1


    i_, j_ = 0,0

    for ind, opk in step_list:    # assumes that these inds are order top to bottom, left to right

        i,j = ind

        # print 'apply op', ind, len(step_list)

        assert(i>=i_), 'step list inds should go from top to bottom'
        if i_== i:  assert(j>=j_), 'step list inds should go from left to right'

        NR, NC  = trotterH.ns[opk]
        op_conn = trotterH.conn[opk]
        ind0    = trotterH.ind0[opk]

        if expH is None:      op  = trotterH.ops[opk]
        else:                 op  = expH[opk]


        try:
            envI = envIs[i]
            envO = envOs[i+NR]

        except(IndexError):
            for ix in range(i_,i):
                b_mpo,err = ENV_SL.get_next_boundary_I(np.conj(pepx_[ix,:]),pepx_[ix,:],envIs[ix],XMAX=XMAX)
                envIs.append(b_mpo)
            envI = envIs[i]
            envO = envOs[i+NR]

            senvRs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,XMAX=XMAX)
                     # list of len L+1
            senvLs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,XMAX=XMAX)
                     # list of len jj+2

        try:
            envL = senvLs[j]
            envR = senvRs[j+NC]
        except(UnboundLocalError):
            # print 'unbound error'

            senvRs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,XMAX=XMAX)
                     # list of len L+1
            senvLs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,XMAX=XMAX)
                     # list of len jj+2
            envL = senvLs[j]
            envR = senvRs[j+NC]

        except(IndexError):
            # print 'index error', i, j

            for jx in range(j_,j):
                jx_ = 2*jx
                e_mpo, err = ENV_SL.get_next_subboundary_L(senvLs[jx],envIs[i][jx_:jx_+2],
                                                           np.conj(pepx_[i:i+NR,jx]),pepx_[i:i+NR,jx],
                                                           envOs[i+NR][jx_:jx_+2],XMAX=XMAX)
                senvLs.append(e_mpo)
            envL = senvLs[j]
            envR = senvRs[j+NC]

        envI_DL = ENV_SL.SL_to_DL_bound(envI,'row')
        envO_DL = ENV_SL.SL_to_DL_bound(envO,'row')
        envL_DL = ENV_SL.SL_to_DL_bound(envL,'col')
        envR_DL = ENV_SL.SL_to_DL_bound(envR,'col')

        # env without lambda on dangling bonds
        lam_i = [pepx_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        bi = ENV_GL.apply_lam_to_boundary_mpo(envI_DL[j:j+NC], lam_i, lam_i)
        lam_o = [pepx_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        bo = ENV_GL.apply_lam_to_boundary_mpo(envO_DL[j:j+NC], lam_o, lam_o)
        lam_l = [pepx_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        bl = ENV_GL.apply_lam_to_boundary_mpo(envL_DL, lam_l, lam_l)
        lam_r = [pepx_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        br = ENV_GL.apply_lam_to_boundary_mpo(envR_DL, lam_r, lam_r)

        sub_pepx = pepx_[i:i+NR,j:j+NC]

        # print [m.shape for idx, m in np.ndenumerate(sub_pepx)]
        # print [m.shape for m in bl]
        # print [m.shape for m in bi]
        # print [m.shape for m in bo]
        # print [m.shape for m in br]

        # print 'trotter io', i, j, opk, trotterH.conn[opk]
        sub_pepx = update_sites_FU(sub_pepx,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,ensure_pos)
        pepx_[i:i+NR,j:j+NC] = sub_pepx
        
        i_,j_ = i,j

    for ix in range(i,i+NR):
        b_mpo,err = ENV_SL.get_next_boundary_I(np.conj(pepx_[ix,:]),pepx_[ix,:],envIs[ix],XMAX=XMAX)
        envIs.append(b_mpo)

    ovlp = ENV_SL.contract_2_bounds(envIs[i+NR],envOs[i+NR],pepx_.lambdas[i+NR-1,:,2],pepx_.lambdas[i+NR-1,:,2],'row')
    norm = np.sqrt( ovlp )

    return pepx_, norm


# @profile
def apply_trotter_layer_LR(pepx, step_list, trotterH, expH=None, DMAX=10, XMAX=100, build_env=False,
                           qr_reduce=False, ensure_pos=False):
    ''' perform trotter step using L and R as main boundaries'''

    pepx_ = pepx.copy()

    envLs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'L', 0, XMAX=XMAX)  # list of len ii+1
    envRs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'R', 0, XMAX=XMAX)  # list of len L+1

    # #############
    # envLsD = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'L', 0, XMAX=XMAX)  # list of len ii+1
    # envRsD = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'R', 0, XMAX=XMAX)  # list of len L+1

    i_, j_ = 0,0

    for ind, opk in step_list:    # assumes that these inds are order top to bottom, left to right

        i,j = ind

        assert(j>=j_), 'step list inds should go from left to right'
        if j_== j:  assert(i>=i_), 'step list inds should go from top to bottom'

        NR, NC  = trotterH.ns[opk]
        op_conn = trotterH.conn[opk]
        ind0    = trotterH.ind0[opk]

        if expH is None:      op  = trotterH.ops[opk]
        else:                 op  = expH[opk]


        try:
            envL = envLs[j]
            envR = envRs[j+NC]

            # #############
            # envLD = envLsD[j]
            # envRD = envRsD[j+NC]

        except(IndexError):
            for jx in range(j_,j):
                b_mpo,err = ENV_SL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLs[jx],XMAX=XMAX)
                envLs.append(b_mpo)
            envL = envLs[j]
            envR = envRs[j+NC]

            senvOs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,XMAX=XMAX)
                     # list of len L+1
            senvIs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,XMAX=XMAX)
                     # list of len jj+2

            # #############
            # for jx in range(j_,j):
            #     b_mpo,err = ENV_GL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLsD[jx],XMAX=XMAX)
            #     envLsD.append(b_mpo)
            # envLD = envLsD[j]
            # envRD = envRsD[j+NC]

            # senvOsD = ENV_GL.get_subboundaries(envLD,envRD,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,XMAX=XMAX)
            #          # list of len L+1
            # senvIsD = ENV_GL.get_subboundaries(envLD,envRD,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,XMAX=XMAX)
            #          # list of len jj+2

        try:
            envI = senvIs[i]
            envO = senvOs[i+NR]

            # #############
            # envID = senvIsD[i]
            # envOD = senvOsD[i+NR]

        except(UnboundLocalError):
            senvOs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,XMAX=XMAX)
                     # list of len L+1
            senvIs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,XMAX=XMAX)
                     # list of len jj+2
            envI = senvIs[i]
            envO = senvOs[i+NR]

            # #############
            # senvOsD = ENV_GL.get_subboundaries(envLD,envRD,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,XMAX=XMAX)
            #          # list of len L+1
            # senvIsD = ENV_GL.get_subboundaries(envLD,envRD,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,XMAX=XMAX)
            #          # list of len jj+2
            # envID = senvIsD[i]
            # envOD = senvOsD[i+NR]

        except(IndexError):
            for ix in range(i_,i):
                ix_ = 2*ix
                e_mpo, err = ENV_SL.get_next_subboundary_I(senvIs[ix],envLs[j][ix_:ix_+2],
                                                           np.conj(pepx_[ix,j:j+NC]),pepx_[ix,j:j+NC],
                                                           envRs[j+NC][ix_:ix_+2],XMAX=XMAX)
                senvIs.append(e_mpo)

            envI = senvIs[i]
            envO = senvOs[i+NR]

            # #############
            # for ix in range(i_,i):
            #     e_mpo, err = ENV_GL.get_next_subboundary_I(senvIsD[ix],envLsD[j][ix],
            #                                                np.conj(pepx_[ix,j:j+NC]),pepx_[ix,j:j+NC],
            #                                                envRsD[j+NC][ix],XMAX=XMAX)
            #     senvIsD.append(e_mpo)
            # envID = senvIsD[i]
            # envOD = senvOsD[i+NR]

        envI_DL = ENV_SL.SL_to_DL_bound(envI,'row')
        envO_DL = ENV_SL.SL_to_DL_bound(envO,'row')
        envL_DL = ENV_SL.SL_to_DL_bound(envL,'col')
        envR_DL = ENV_SL.SL_to_DL_bound(envR,'col')

        # print 'envI', MPX.norm( MPX.add(envID, MPX.mul(-1,envI_DL)) )
        # print 'envO', MPX.norm( MPX.add(envOD, MPX.mul(-1,envO_DL)) )
        # print 'envL', MPX.norm( MPX.add(envLD, MPX.mul(-1,envL_DL)) )
        # print 'envR', MPX.norm( MPX.add(envRD, MPX.mul(-1,envR_DL)) )

        # env without lambda on dangling bonds
        lam_i = [pepx_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        bi = ENV_GL.apply_lam_to_boundary_mpo(envI_DL, lam_i, lam_i)
        lam_o = [pepx_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        bo = ENV_GL.apply_lam_to_boundary_mpo(envO_DL, lam_o, lam_o)
        lam_l = [pepx_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        bl = ENV_GL.apply_lam_to_boundary_mpo(envL_DL[i:i+NR], lam_l, lam_l)
        lam_r = [pepx_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        br = ENV_GL.apply_lam_to_boundary_mpo(envR_DL[i:i+NR], lam_r, lam_r)


        # #############
        # # env without lambda on dangling bonds
        # lam_i = [pepx_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        # bid = ENV_GL.apply_lam_to_boundary_mpo(envID, lam_i, lam_i)
        # lam_o = [pepx_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        # bod = ENV_GL.apply_lam_to_boundary_mpo(envOD, lam_o, lam_o)
        # lam_l = [pepx_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        # bld = ENV_GL.apply_lam_to_boundary_mpo(envLD[i:i+NR], lam_l, lam_l)
        # lam_r = [pepx_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        # brd = ENV_GL.apply_lam_to_boundary_mpo(envRD[i:i+NR], lam_r, lam_r)

        # print 'envIb', MPX.norm( MPX.add(bid, MPX.mul(-1,bi)) )
        # print 'envOb', MPX.norm( MPX.add(bod, MPX.mul(-1,bo)) )
        # print 'envLb', MPX.norm( MPX.add(bld, MPX.mul(-1,bl)) )
        # print 'envRb', MPX.norm( MPX.add(brd, MPX.mul(-1,br)) )


        sub_pepx = pepx_[i:i+NR,j:j+NC]

        # print [m.shape for idx, m in np.ndenumerate(sub_pepx)]
        # print [m.shape for m in bl]
        # print [m.shape for m in bi]
        # print [m.shape for m in bo]
        # print [m.shape for m in br]

        sub_pepx = update_sites_FU(sub_pepx,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,ensure_pos)
        pepx_[i:i+NR,j:j+NC] = sub_pepx

        i_,j_ = i,j


    for jx in range(j,j+NC):
        b_mpo,err = ENV_SL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLs[jx],XMAX=XMAX)
        envLs.append(b_mpo)

    ovlp = ENV_SL.contract_2_bounds(envLs[j+NC],envRs[j+NC],pepx_.lambdas[:,j+NC-1,3],pepx_.lambdas[:,j+NC-1,3],'col')
    norm = np.sqrt( ovlp )

    # for jx in range(i,pepx.shape[1]):
    #     print 'getting envs', jx
    #     b_mpo,err = ENV_GL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLs[jx],XMAX=XMAX)
    #     envLs.append(b_mpo)

    # norm = np.sqrt( ENV_GL.ovlp_from_bound(envLs[pepx.shape[1]]) )

    # print 'LR', norm, norm1
        
    return pepx_, norm


### perfrom trotter step TE ###
# @profile
def run_TE(dt,totT,trotterH,initPEPX,obs_pepos=None,DMAX=10,XMAX=100,normalize=True,pepx_type='peps',
              te_type='SU',build_env=False,qr_reduce=False,ensure_pos=False,trotter_version=2):
    ''' time evolution algorithm
        dt, totT = time step, total amounth of time
        trotterH = [list of trotter steps generating H, corresponding orientations 'h','v','hv']
                   len(orientation) matches len(trotter mpo).   eg. 'hv' = diagonal, 'hh' = horizontal over 2 sites
        trotterH = dictionary of trotter steps
                   key = orientation 'h', 'v', 'hv', 
                   ... length of key = len(trotter mpo), 'hv' = diagonal, 'hh' = horizontal over 2 sites
                   mapped to MPO
        initPEPX = initial thermal PEPO
    '''

    L1,L2 = initPEPX.shape
    
    exp_tsteps = get_exp(trotterH,dt,pepx_type)

    # if obs_pepo is None:  obs_pepo = Op.PEPO_mag((L1,L2),'SZ').getPEPO()


    # start time evolution via trotter stpes
    pepx = initPEPX.copy()
    dbs  = pepx.phys_bonds.copy()

    tstep = 0
    numSteps = int(abs(totT/dt))
    obs_t = {}
    for obs_key in obs_pepos.keys():
        # obs = PEPX_GL.meas_obs(tldm, obs_pepos[obs_key], XMAX=100, return_norm=False)
        obs_t[obs_key] = []


    step_lists = get_trotter_layers(trotterH,version=trotter_version)

    while tstep < numSteps:

        if pepx_type in ['DM','pepo','rho']:   pepx_ = PEPX_GL.flatten(pepx)
        else:                                  pepx_ = pepx


        for xdir, step_list in step_lists:

            if te_type in ['FU','full']:
                if xdir in ['io','oi']:
                    pepx_, norm = apply_trotter_layer_IO(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,build_env,
                                                         qr_reduce,ensure_pos)
                elif xdir in ['lr','rl']:
                    pepx_, norm = apply_trotter_layer_LR(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,build_env,
                                                         qr_reduce,ensure_pos)

                pepx_ = PEPX_GL.mul(1./norm, pepx_)
                # print 'fu norm', norm

            elif te_type in ['SU','simple']:
                pepx_ = apply_trotter_layer_SU(pepx_,step_list,trotterH,exp_tsteps,DMAX)

                # normalize_step = False
                # if normalize_step:
                #     if pepx_type in ['DM','pepo','rho']:
                #         pepx = PEPX_GL.unflatten(pepx_,dbs)
                #         norm = PEPX_GL.trace_norm(pepx,XMAX=XMAX)
                #     else: 
                #         norm = PEPX_GL.norm(pepx_,XMAX=XMAX)

                #     pepx_ = PEPX_GL.mul(1./norm, pepx_)



        if pepx_type in ['DM','pepo','rho']:   pepx = PEPX_GL.unflatten(pepx_,dbs)
        else:                                  pepx = pepx_


        # measure observable
        for obs_key in obs_pepos.keys():
            # temp = PEPX.meas_obs(beta_peps, obs_pepos[obs])
            # obs_steps[obs] += [PEPX.meas_obs(beta_peps, obs_pepos[obs_key])]
            obs, norm = PEPX_GL.meas_obs(pepx, obs_pepos[obs_key], XMAX=100, return_norm=True)
            obs_t[obs_key] += [obs]

        try:
            obs = obs_t['H'][-1]
            obs_key = 'H'
        except:  pass

        if isinstance(obs,np.float) or isinstance(obs,np.complex):    print 'obs', tstep, obs_key, obs/L1/L2


        ## ensure canonicalization if possible
        # print 'check time step'
        if L1 > 1 and L2 == 1:
            gs,ls,axT = PEPX_GL.get_sites( pepx, (0,0), 'o'*(L1-1))
            # PEPX_GL.check_GL_canonical(gs,ls)

            gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=0)
            # print 'check canon'
            PEPX_GL.check_GL_canonical(gs_,ls_)
            pepx = PEPX_GL.set_sites( pepx, (0,0), 'o'*(L1-1), gs_, ls_, axT)

        if L1 == 1 and L2 > 1:
            gs,ls,axT = PEPX_GL.get_sites( pepx, (0,0), 'r'*(L2-1))
            # PEPX_GL.check_GL_canonical(gs,ls)

            gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=1)
            # print 'norm',PEPX_GL.norm(tldm)
            PEPX_GL.check_GL_canonical(gs_,ls_)
            pepx = PEPX_GL.set_sites( pepx, (0,0), 'r'*(L2-1), gs_, ls_, axT)
        # print 'end check time step'


        tstep += 1

        ## save peps
        if False:  
            np.save('peps_data/runtime_peps_L%dx%d_D%d_gammas.npy' %(L1,L2,DMAX),pepx.view(np.ndarray))
            np.save('peps_data/runtime_peps_L%dx%d_D%d_lambdas.npy'%(L1,L2,DMAX),pepx.lambdas)
        
    return obs_t, pepx

### 



def get_trotter_layers(trotterH,version=0):

    return PXTE.get_trotter_layers(trotterH,version)
