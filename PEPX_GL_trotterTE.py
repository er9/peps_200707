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
# import PEPS_GL_env_lam as ENV_GLL
# import PEPS_GL_env_lam_SL as ENV_SL
import PEPS_GL_env_nolam_SL as ENV_SL
import Operator_2D as Op
import TimeEvolution as TE

# import PEPX_trotterTE as PXTE


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



##################################
#### update sites via SU or FU ###
##################################

def update_sites_SU(pepx,ind0,op_conn,op,DMAX=10):

    gam, lam, axT_invs = PEPX_GL.get_sites(pepx,ind0,op_conn)
    gam_, lam_, errs = PEPS_GL.simple_update(gam, lam, op, DMAX=DMAX, direction=0, normalize=True)
    # gam_, lam_ = PEPX_GL.regularize_GL_list(gam_,lam_)
    pepx_ = PEPX_GL.set_sites(pepx,ind0,op_conn,gam_,lam_,axT_invs)

    return pepx_


def update_sites_FU(pepx,ind0,op_conn,op,envs_list,DMAX=10,XMAX=100,build_env=False,qr_reduce=False,
                    ensure_pos=False, alsq_2site=False):

    xs,ys = PEPX.get_conn_inds(op_conn,ind0)

    pepx_ = PEPS_GL.alsq_block_update(pepx,op_conn,[xs,ys],envs_list,op,DMAX=DMAX,XMAX=XMAX,site_idx=None,
                                      normalize=False,build_env=build_env,qr_reduce=qr_reduce,
                                      ensure_pos=ensure_pos,alsq_2site=alsq_2site) #,init_guess=pepx)
    return pepx_
    


##################################
#### SIMPLE UPDATE TROTTER    ####
##################################

# @profile
def apply_trotter_layer_SU(pepx, step_list, trotterH, expH=None, DMAX=10):

    pepx_ = pepx.copy()
    L1,L2 = pepx.shape

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

        # print 'trotter SU step norm', PEPX_GL.norm(pepx_)

        # print 'check trotter step SU', ind, opk
        if L1 > 1 and L2 == 1: # and NR==2:
            gs,ls,axT = PEPX_GL.get_sites( pepx_, (0,0), 'o'*(L1-1))
            PEPX_GL.check_GL_canonical(gs[:i+NR-1],ls[:i+NR], check='L')
            PEPX_GL.check_GL_canonical(gs[i+NR-1:],ls[i+NR-1:], check='R')
        # print 'end check trotter step'

    return pepx_


#####################################
#### DOUBLE LAYER CONTRACTION FU ####
#####################################

# @profile
def apply_trotter_layer_IO(pepx, step_list, trotterH, expH=None, DMAX=10, XMAX=100, build_env=False, 
                           qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using I and O as main boundaries'''

    pepx_ = pepx.copy()

    envIs = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'I', 0, XMAX=XMAX)  # list of len ii+1
    envOs = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'O', 0, XMAX=XMAX)  # list of len L+1


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
                b_mpo,err = ENV_GL.get_next_boundary_I(np.conj(pepx_[ix,:]),pepx_[ix,:],envIs[ix],XMAX=XMAX)
                envIs.append(b_mpo)

            # temp_mpo, err = ENV_GL.get_next_boundary_I(np.conj(pepx_[i,:]),pepx_[i,:],envIs[i],XMAX=XMAX)
            # temp_ovlp = ENV_GL.contract_2_bounds(temp_mpo,envOs[i+1],
            #                                  pepx_.lambdas[i+1,:,1],pepx_.lambdas[i+1,:,1])
            # print 'new boundI --> norm', i, len(envIs), temp_ovlp

            envI = envIs[i]
            envO = envOs[i+NR]

            senvRs = ENV_GL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,XMAX=XMAX)
                     # list of len L+1
            senvLs = ENV_GL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,XMAX=XMAX)
                     # list of len jj+2

        try:
            envL = senvLs[j]
            envR = senvRs[j+NC]
        except(UnboundLocalError):
            # print 'unbound error'

            senvRs = ENV_GL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,XMAX=XMAX)
                     # list of len L+1
            senvLs = ENV_GL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,XMAX=XMAX)
                     # list of len jj+2
            envL = senvLs[j]
            envR = senvRs[j+NC]

        except(IndexError):

            # print 'index error', i, j

            for jx in range(j_,j):
                e_mpo, err = ENV_GL.get_next_subboundary_L(senvLs[jx],envIs[i][jx],
                                                           np.conj(pepx_[i:i+NR,jx]),pepx_[i:i+NR,jx],
                                                           envOs[i+NR][jx],XMAX=XMAX)
                senvLs.append(e_mpo)
            envL = senvLs[j]
            envR = senvRs[j+NC]


        # # env without lambda on dangling bonds
        # lam_i = [pepx_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        # bi = ENV_GL.apply_lam_to_boundary_mpo(envI[j:j+NC], lam_i, lam_i)
        # lam_o = [pepx_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        # bo = ENV_GL.apply_lam_to_boundary_mpo(envO[j:j+NC], lam_o, lam_o)
        # lam_l = [pepx_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        # bl = ENV_GL.apply_lam_to_boundary_mpo(envL, lam_l, lam_l)
        # lam_r = [pepx_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        # br = ENV_GL.apply_lam_to_boundary_mpo(envR, lam_r, lam_r)

        bi = envI[j:j+NC]
        bo = envO[j:j+NC]
        bl = envL
        br = envR

        sub_pepx = pepx_[i:i+NR,j:j+NC]

        # print 'trotter io', i, j, opk, trotterH.conn[opk]
        sub_pepx = update_sites_FU(sub_pepx,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        pepx_[i:i+NR,j:j+NC] = sub_pepx
        
        # print 'updated norm', ENV_GL.embed_sites_norm( sub_pepx, [bl,bi,bo,br] )

        i_,j_ = i,j



    for ix in range(i,i+NR):
        b_mpo,err = ENV_GL.get_next_boundary_I(np.conj(pepx_[ix,:]),pepx_[ix,:],envIs[ix],XMAX=XMAX)
        envIs.append(b_mpo)

    ovlp = ENV_GL.contract_2_bounds( envIs[i+NR], envOs[i+NR], pepx_.lambdas[i+NR-1,:,2], pepx_.lambdas[i+NR-1,:,2] )
    norm = np.sqrt( ovlp )

    # print 'final ovlp norm', norm

    return pepx_, norm


# @profile
def apply_trotter_layer_LR(pepx, step_list, trotterH, expH=None, DMAX=10, XMAX=100, build_env=False,
                           qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using L and R as main boundaries'''

    pepx_ = pepx.copy()

    envLs = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'L', 0, XMAX=XMAX)  # list of len ii+1
    envRs = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'R', 0, XMAX=XMAX)  # list of len L+1

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

        except(IndexError):
            for jx in range(j_,j):
                b_mpo,err = ENV_GL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLs[jx],XMAX=XMAX)
                envLs.append(b_mpo)
            envL = envLs[j]
            envR = envRs[j+NC]

            senvOs = ENV_GL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,XMAX=XMAX)
                     # list of len L+1
            senvIs = ENV_GL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,XMAX=XMAX)
                     # list of len jj+2

        try:
            envI = senvIs[i]
            envO = senvOs[i+NR]

        except(UnboundLocalError):
            senvOs = ENV_GL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,XMAX=XMAX)
                     # list of len L+1
            senvIs = ENV_GL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,XMAX=XMAX)
                     # list of len jj+2
            envI = senvIs[i]
            envO = senvOs[i+NR]

        except(IndexError):
            for ix in range(i_,i):
                e_mpo, err = ENV_GL.get_next_subboundary_I(senvIs[ix],envLs[j][ix],
                                                           np.conj(pepx_[ix,j:j+NC]),pepx_[ix,j:j+NC],
                                                           envRs[j+NC][ix],XMAX=XMAX)
                senvIs.append(e_mpo)
            envI = senvIs[i]
            envO = senvOs[i+NR]


        # # env without lambda on dangling bonds
        # lam_i = [pepx_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        # bi = ENV_GL.apply_lam_to_boundary_mpo(envI, lam_i, lam_i)
        # lam_o = [pepx_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        # bo = ENV_GL.apply_lam_to_boundary_mpo(envO, lam_o, lam_o)
        # lam_l = [pepx_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        # bl = ENV_GL.apply_lam_to_boundary_mpo(envL[i:i+NR], lam_l, lam_l)
        # lam_r = [pepx_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        # br = ENV_GL.apply_lam_to_boundary_mpo(envR[i:i+NR], lam_r, lam_r)

        bi = envI
        bo = envO
        bl = envL[i:i+NR]
        br = envR[i:i+NR]

        sub_pepx = pepx_[i:i+NR,j:j+NC]
        sub_pepx = update_sites_FU(sub_pepx,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        pepx_[i:i+NR,j:j+NC] = sub_pepx

        i_,j_ = i,j


    for jx in range(j,j+NC):
        b_mpo,err = ENV_GL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLs[jx],XMAX=XMAX)
        envLs.append(b_mpo)

    ovlp = ENV_GL.contract_2_bounds( envLs[j+NC], envRs[j+NC], pepx_.lambdas[:,j+NC-1,3], pepx_.lambdas[:,j+NC-1,3] )
    norm = np.sqrt( ovlp )

    return pepx_, norm


#####################################
#### SINGLE LAYER CONTRACTION FU ####
#####################################

# @profile
def apply_trotter_layer_IO_SL(pepx, step_list, trotterH, expH=None, DMAX=10, XMAX=100, scaleX=1, build_env=False, 
                              qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using I and O as main boundaries'''

    pepx_ = pepx.copy()

    envIs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'I', 0, XMAX=XMAX, scaleX=scaleX)  # list of len ii+1
    envOs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'O', 0, XMAX=XMAX, scaleX=scaleX)  # list of len L+1

    # #############
    # envIsD = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'I', 0, XMAX=XMAX)  # list of len ii+1
    # envOsD = ENV_GL.get_boundaries( np.conj(pepx_), pepx_, 'O', 0, XMAX=XMAX)  # list of len L+1


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

            # #############
            # envID = envIsD[i]
            # envOD = envOsD[i+NR]

        except(IndexError):
            for ix in range(i_,i):
                b_mpo,err = ENV_SL.get_next_boundary_I(np.conj(pepx_[ix,:]),pepx_[ix,:],envIs[ix],
                                                       XMAX=XMAX,scaleX=scaleX)
                envIs.append(b_mpo)

            # temp_mpo, err = ENV_SL.get_next_boundary_I(np.conj(pepx_[i,:]),pepx_[i,:],envIs[i],XMAX=XMAX)
            # temp_ovlp = ENV_SL.contract_2_bounds(temp_mpo,envOs[i+1],
            #                                  pepx_.lambdas[i+1,:,1],pepx_.lambdas[i+1,:,1], 'row')
            # print 'new boundI --> norm', i, len(envIs), temp_ovlp

            envI = envIs[i]
            envO = envOs[i+NR]

            senvRs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvLs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len jj+2

            # #############
            # for ix in range(i_,i):
            #     b_mpo,err = ENV_GL.get_next_boundary_I(np.conj(pepx_[ix,:]),pepx_[ix,:],envIsD[ix],XMAX=XMAX)
            #     envIsD.append(b_mpo)
            # envID = envIsD[i]
            # envOD = envOsD[i+NR]

            # senvRsD = ENV_GL.get_subboundaries(envID,envOD,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,XMAX=XMAX)
            #          # list of len L+1
            # senvLsD = ENV_GL.get_subboundaries(envID,envOD,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,XMAX=XMAX)
            #          # list of len jj+2

        try:
            envL = senvLs[j]
            envR = senvRs[j+NC]

            # #############
            # envLD = senvLsD[j]
            # envRD = senvRsD[j+NC]

        except(UnboundLocalError):
            # print 'unbound error'

            senvRs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvLs = ENV_SL.get_subboundaries(envI,envO,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len jj+2
            envL = senvLs[j]
            envR = senvRs[j+NC]

            # #############
            # senvRsD = ENV_GL.get_subboundaries(envID,envOD,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'R',0,XMAX=XMAX)
            #          # list of len L+1
            # senvLsD = ENV_GL.get_subboundaries(envID,envOD,np.conj(pepx_[i:i+NR,:]),pepx_[i:i+NR,:],'L',j,XMAX=XMAX)
            #          # list of len jj+2
            # envLD = senvLsD[j]
            # envRD = senvRsD[j+NC]

        except(IndexError):
            # print 'index error', i, j

            for jx in range(j_,j):
                jx_ = 2*jx
                e_mpo, err = ENV_SL.get_next_subboundary_L(senvLs[jx],envIs[i][jx_:jx_+2],
                                                           np.conj(pepx_[i:i+NR,jx]),pepx_[i:i+NR,jx],
                                                           envOs[i+NR][jx_:jx_+2],XMAX=XMAX,scaleX=scaleX)
                senvLs.append(e_mpo)
            envL = senvLs[j]
            envR = senvRs[j+NC]

            # #############
            # for jx in range(j_,j):
            #     e_mpo, err = ENV_GL.get_next_subboundary_L(senvLsD[jx],envIsD[i][jx],
            #                                                np.conj(pepx_[i:i+NR,jx]),pepx_[i:i+NR,jx],
            #                                                envOsD[i+NR][jx],XMAX=XMAX)
            #     senvLsD.append(e_mpo)
            # envLD = senvLsD[j]
            # envRD = senvRsD[j+NC]

        envI_DL = ENV_SL.SL_to_DL_bound(envI,'row')
        envO_DL = ENV_SL.SL_to_DL_bound(envO,'row')
        envL_DL = ENV_SL.SL_to_DL_bound(envL,'col')
        envR_DL = ENV_SL.SL_to_DL_bound(envR,'col')

        # # env without lambda on dangling bonds
        # lam_i = pepx_.lambdas[i,j:j+NC,1]
        # bi = ENV_GLL.apply_lam_to_boundary_mpo(envI_DL[j:j+NC], lam_i, lam_i)
        # lam_o = pepx_.lambdas[i+NR-1,j:j+NC,2]
        # bo = ENV_GLL.apply_lam_to_boundary_mpo(envO_DL[j:j+NC], lam_o, lam_o)
        # lam_l = pepx_.lambdas[i:i+NR,j,0]
        # bl = ENV_GLL.apply_lam_to_boundary_mpo(envL_DL, lam_l, lam_l)
        # lam_r = pepx_.lambdas[i:i+NR,j+NC-1,3]
        # br = ENV_GLL.apply_lam_to_boundary_mpo(envR_DL, lam_r, lam_r)

        bi = envI_DL[j:j+NC]
        bo = envO_DL[j:j+NC]
        bl = envL_DL
        br = envR_DL

        # #############
        # env_SL = ENV_GL.build_env([bl,bi,bo,br])

        # biD = envID[j:j+NC]
        # boD = envOD[j:j+NC]
        # blD = envLD
        # brD = envRD
        # env_DL = ENV_GL.build_env([blD,biD,boD,brD])
        # env_err = np.linalg.norm(env_SL-env_DL)
        # # if env_err > 1.0e-5:
        # #     print 'warning: DL vs SL IO env difference is large', env_err
        # if env_err > 1.0e-1:
        #     print 'IO env err', i,j,env_err
        #     print [np.linalg.norm(m) for idx,m in np.ndenumerate(pepx_)]
        #     # print [pepx_.lambdas[idx] for idx in np.ndindex(pepx_.shape)]
        #     # print PEPX_GL.norm(pepx_,XMAX=XMAX)
        #     # print [np.linalg.norm(m) for m in biD], [np.linalg.norm(m) for m in bi]
        #     # print [np.linalg.norm(m) for m in boD], [np.linalg.norm(m) for m in bo]
        #     # print [np.linalg.norm(m) for m in blD], [np.linalg.norm(m) for m in bl]
        #     # print [np.linalg.norm(m) for m in brD], [np.linalg.norm(m) for m in br]
        #     # exit()
 

        sub_pepx = pepx_[i:i+NR,j:j+NC]
        sub_pepx = update_sites_FU(sub_pepx,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        pepx_[i:i+NR,j:j+NC] = sub_pepx
        
        # print 'updated norm', ENV_GL.embed_sites_norm( sub_pepx, [bl,bi,bo,br] )

        i_,j_ = i,j

    for ix in range(i,i+NR):
        b_mpo,err = ENV_SL.get_next_boundary_I(np.conj(pepx_[ix,:]),pepx_[ix,:],envIs[ix],XMAX=XMAX,scaleX=scaleX)
        envIs.append(b_mpo)

    ovlp = ENV_SL.contract_2_bounds(envIs[i+NR],envOs[i+NR],pepx_.lambdas[i+NR-1,:,2],pepx_.lambdas[i+NR-1,:,2],'row')
    norm = np.sqrt( ovlp )
    # print 'final ovlp norm', norm

    return pepx_, norm


# @profile
def apply_trotter_layer_LR_SL(pepx, step_list, trotterH, expH=None, DMAX=10, XMAX=100, scaleX=1, build_env=False,
                              qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using L and R as main boundaries'''

    pepx_ = pepx.copy()

    envLs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'L', 0, XMAX=XMAX, scaleX=scaleX)  # list of len ii+1
    envRs = ENV_SL.get_boundaries( np.conj(pepx_), pepx_, 'R', 0, XMAX=XMAX, scaleX=scaleX)  # list of len L+1

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
                b_mpo,err = ENV_SL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLs[jx],
                                                       XMAX=XMAX,scaleX=scaleX)
                envLs.append(b_mpo)
            envL = envLs[j]
            envR = envRs[j+NC]

            senvOs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvIs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,
                                              XMAX=XMAX,scaleX=scaleX)
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
            senvOs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'O',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvIs = ENV_SL.get_subboundaries(envL,envR,np.conj(pepx_[:,j:j+NC]),pepx_[:,j:j+NC],'I',i,
                                              XMAX=XMAX,scaleX=scaleX)
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
                                                           envRs[j+NC][ix_:ix_+2],XMAX=XMAX,scaleX=scaleX)
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

        # # env without lambda on dangling bonds
        # lam_i = pepx_.lambdas[i,j:j+NC,1]
        # bi = ENV_GLL.apply_lam_to_boundary_mpo(envI_DL, lam_i, lam_i)
        # lam_o = pepx_.lambdas[i+NR-1,j:j+NC,2]
        # bo = ENV_GLL.apply_lam_to_boundary_mpo(envO_DL, lam_o, lam_o)
        # lam_l = pepx_.lambdas[i:i+NR,j,0]
        # bl = ENV_GLL.apply_lam_to_boundary_mpo(envL_DL[i:i+NR], lam_l, lam_l)
        # lam_r = pepx_.lambdas[i:i+NR,j+NC-1,3]
        # br = ENV_GLL.apply_lam_to_boundary_mpo(envR_DL[i:i+NR], lam_r, lam_r)

        bi = envI_DL
        bo = envO_DL
        bl = envL_DL[i:i+NR]
        br = envR_DL[i:i+NR]

        # #############
        # env_SL = ENV_GL.build_env([bl,bi,bo,br])

        # biD = envID
        # boD = envOD
        # blD = envLD[i:i+NR]
        # brD = envRD[i:i+NR]
        # env_DL = ENV_GL.build_env([blD,biD,boD,brD])
        # env_err = np.linalg.norm( env_DL - env_SL )
        # # if env_err > 1.0e-5:
        # #     print 'warning:  DL vs SL LR env norm diff bad', env_err
        # if env_err > 1.0e-1:
        #     print 'LR env err', i,j, env_err
        #     print [np.linalg.norm(m) for idx,m in np.ndenumerate(pepx_)]
        #     # print [pepx_.lambdas[idx] for idx in np.ndindex(pepx_.shape)]
        #     # print PEPX_GL.norm(pepx_,XMAX=XMAX)
        #     # print [np.linalg.norm(m) for m in biD], [np.linalg.norm(m) for m in bi]
        #     # print [np.linalg.norm(m) for m in boD], [np.linalg.norm(m) for m in bo]
        #     # print [np.linalg.norm(m) for m in blD], [np.linalg.norm(m) for m in bl]
        #     # print [np.linalg.norm(m) for m in brD], [np.linalg.norm(m) for m in br]
        #     # exit()


        sub_pepx = pepx_[i:i+NR,j:j+NC]
        sub_pepx = update_sites_FU(sub_pepx,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        pepx_[i:i+NR,j:j+NC] = sub_pepx

        i_,j_ = i,j


    for jx in range(j,j+NC):
        b_mpo,err = ENV_SL.get_next_boundary_L(np.conj(pepx_[:,jx]),pepx_[:,jx],envLs[jx],XMAX=XMAX,scaleX=scaleX)
        envLs.append(b_mpo)

    ovlp = ENV_SL.contract_2_bounds(envLs[j+NC],envRs[j+NC],pepx_.lambdas[:,j+NC-1,3],pepx_.lambdas[:,j+NC-1,3],'col')
    norm = np.sqrt( ovlp )

    return pepx_, norm


###############################
### perfrom trotter step TE ###
###############################

# @profile
def run_TE(dt,totT,trotterH,initPEPX,obs_pepos=None,DMAX=10,XMAX=100,XMAX2=100,normalize=True,pepx_type='peps',
           te_type='SU',build_env=False,qr_reduce=False,ensure_pos=False,trotter_version=2,contract_SL=True,
           scaleX=1,truncate_run=False,alsq_2site=False,run_tol=1.e-6):
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
    
    time1 = time.time()
    exp_tsteps = get_exp(trotterH,dt,pepx_type)


    # start time evolution via trotter stpes
    pepx = initPEPX.copy()
    dbs  = pepx.phys_bonds.copy()

    tstep = 0
    numSteps = int(round(abs(totT/dt),2))
    obs_t = {}
    for obs_key in obs_pepos.keys():
        obs_t[obs_key] = []


    step_lists = get_trotter_layers(trotterH,version=trotter_version)

    while tstep < numSteps:

        if pepx_type in ['DM','pepo','rho']:   pepx_ = PEPX_GL.flatten(pepx)
        else:                                  pepx_ = pepx.copy()

        # print 'check time step full init'
        # if L1 > 1 and L2 == 1:
        #     gs,ls,axT = PEPX_GL.get_sites( pepx, (0,0), 'o'*(L1-1))
        #     PEPX_GL.check_GL_canonical(gs,ls,check='RL')


        for xdir, step_list in step_lists:

            if te_type in ['FU','full']:
                if contract_SL:
                    if xdir in ['io','oi']:
                        pepx_, norm = apply_trotter_layer_IO_SL(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,scaleX,
                                                             build_env,qr_reduce,ensure_pos,alsq_2site)
                    elif xdir in ['lr','rl']:
                        pepx_, norm = apply_trotter_layer_LR_SL(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,scaleX,
                                                             build_env,qr_reduce,ensure_pos,alsq_2site)
                else:
                    if xdir in ['io','oi']:
                        pepx_, norm = apply_trotter_layer_IO(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                             build_env,qr_reduce,ensure_pos,alsq_2site)
                    elif xdir in ['lr','rl']:
                        pepx_, norm = apply_trotter_layer_LR(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                             build_env,qr_reduce,ensure_pos,alsq_2site)

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
        else:                                  pepx = pepx_.copy()


        # # need this for 2D where things don't really stay canonical
        # if te_type in ['SU','simple','FU']:
        #     norm = PEPX_GL.norm(pepx,XMAX=XMAX2)
        #     pepx = PEPX_GL.mul(1./norm,pepx)
        #     # print 'tstep norm', norm

        # measure observable
        for obs_key in obs_pepos.keys():
            obs,norm = PEPX_GL.meas_obs(pepx,obs_pepos[obs_key],XMAX=XMAX2,return_norm=True,contract_SL=contract_SL)
            pepx = PEPX_GL.mul(1./norm,pepx)    ## obs is already normalized

            # obs = PEPX_GL.meas_obs(pepx,obs_pepos[obs_key],XMAX=XMAX2,return_norm=False,
            #                        contract_SL=contract_SL,scaleX=scaleX)
            obs_t[obs_key] += [obs]
            print 'obs norm', norm

        try:
            obs = obs_t['H'][-1]
            obs_key = 'H'
            H_diff = obs - np.min(obs_t['H'])
        except:
            H_diff = -1

        try:
            if isinstance(obs,np.float) or isinstance(obs,np.complex):    print 'obs', tstep, obs_key, obs/L1/L2
        except(UnboundLocalError):
            # print 'obs', tstep    # no obs defined
            pass

        if truncate_run and (H_diff > 1.0e-6 or np.abs(H_diff) < run_tol):
            print 'energy went up or within run tolerance; truncating TE run', H_diff
            break

        ## ensure canonicalization if possible
        # print 'check time step full', tstep
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

        # ## save peps
        # if False:  
        #     np.save('peps_data/runtime_peps_L%dx%d_D%d_gammas.npy' %(L1,L2,DMAX),pepx.view(np.ndarray))
        #     np.save('peps_data/runtime_peps_L%dx%d_D%d_lambdas.npy'%(L1,L2,DMAX),pepx.lambdas)
        

    return obs_t, pepx



def run_TE_listH(dt,totT,trotterHs,initPEPX,obs_pepos=None,DMAXs=[10],XMAXs=[100],XMAX2s=[100],te_types=['SU'],
                 use_exps=[True],normalize=True,build_env=False,qr_reduce=False,ensure_pos=False,trotter_version=2,
                 contract_SL=True,truncate_run=False,run_tol=1.e-6):

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
    pepx = initPEPX.copy()
    dbs  = pepx.phys_bonds.copy()
    
    time1 = time.time()


    obs_t = {}
    for obs_key in obs_pepos.keys():
        obs_t[obs_key] = []

    exp_tsteps_list = []
    for i in range(len(trotterHs)):
        if use_exps[i%len(use_exps)]:     exp_tsteps_list += [ get_exp(trotterHs[i],dt,'peps') ]
        else:                             exp_tsteps_list += [None]

    step_lists_list = [ get_trotter_layers(trotterH,version=trotter_version) for trotterH in trotterHs ]


    # start time evolution via trotter stpes
    tstep = 0
    numSteps = int(abs(totT/dt))

    while tstep < numSteps:

        pepx_ = pepx.copy()

        step_ind = 0 
        for step_lists in step_lists_list:

            DMAX  = DMAXs [step_ind%len(DMAXs)]
            XMAX  = XMAXs [step_ind%len(XMAXs)]
            XMAX2 = XMAX2s[step_ind%len(XMAX2s)]
            te_type = te_types[step_ind%len(te_types)]

            print step_ind, te_type, type(exp_tsteps_list[step_ind%len(exp_tsteps_list)]) #use_exps[step_ind%len(use_exps)]

            exp_tsteps = exp_tsteps_list[step_ind]
            trotterH = trotterHs[step_ind]
            # print 'trotter', step_ind

            for xdir, step_list in step_lists:

                if te_type in ['FU','full']:
                    if contract_SL:
                        if xdir in ['io','oi']:
                            pepx_, norm = apply_trotter_layer_IO_SL(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 qr_reduce=qr_reduce,ensure_pos=ensure_pos)
                        elif xdir in ['lr','rl']:
                            pepx_, norm = apply_trotter_layer_LR_SL(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 qr_reduce=qr_reduce,ensure_pos=ensure_pos)
                    else:
                        if xdir in ['io','oi']:
                            pepx_, norm = apply_trotter_layer_IO(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 build_env,qr_reduce,ensure_pos)
                        elif xdir in ['lr','rl']:
                            pepx_, norm = apply_trotter_layer_LR(pepx_,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 build_env,qr_reduce,ensure_pos)

                    pepx_ = PEPX_GL.mul(1./norm, pepx_)
                    # print 'fu norm', norm

                elif te_type in ['SU','simple']:

                    pepx_ = apply_trotter_layer_SU(pepx_,step_list,trotterH,exp_tsteps,DMAX)

            step_ind += 1

        pepx = pepx_.copy()

        # # need this for 2D where things don't really stay canonical
        # if te_type in ['SU','simple','FU']:
        #     norm = PEPX_GL.norm(pepx,XMAX=XMAX2)
        #     pepx = PEPX_GL.mul(1./norm,pepx)
        #     # print 'tstep norm', norm

        # measure observable
        for obs_key in obs_pepos.keys():
            obs,norm = PEPX_GL.meas_obs(pepx,obs_pepos[obs_key],XMAX=XMAX2,return_norm=True,contract_SL=contract_SL)
            pepx = PEPX_GL.mul(1./norm,pepx)    ## obs is already normalized

            # obs = PEPX_GL.meas_obs(pepx,obs_pepos[obs_key],XMAX=XMAX2,return_norm=False,
            #                        contract_SL=contract_SL,scaleX=scaleX)
            obs_t[obs_key] += [obs]
            print 'obs norm', norm

        try:
            obs = obs_t['H'][-1]
            obs_key = 'H'
            H_diff = obs - np.min(obs_t['H'])
        except:
            H_diff = -1

        try:
            if isinstance(obs,np.float) or isinstance(obs,np.complex):    print 'obs', tstep, obs_key, obs/L1/L2
        except(UnboundLocalError):
            # print 'obs', tstep    # no obs defined
            pass

        if truncate_run and (H_diff > 1.0e-6 or np.abs(H_diff) < run_tol):
            print 'energy went up or within run tolerance; truncating TE run', H_diff
            break

        ## ensure canonicalization if possible
        # print 'check time step full', tstep
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

        # ## save peps
        # if False:  
        #     np.save('peps_data/runtime_peps_L%dx%d_D%d_gammas.npy' %(L1,L2,DMAX),pepx.view(np.ndarray))
        #     np.save('peps_data/runtime_peps_L%dx%d_D%d_lambdas.npy'%(L1,L2,DMAX),pepx.lambdas)
        

    return obs_t, pepx


## convert trotterH into trotter layers
def get_trotter_layers(trotterH,version=0):

    L1,L2 = trotterH.Ls

    ### trotter steps on even and then odd bonds.
    if version == 0:

        NR, NC = trotterH.it_sh
 
        step_lists = []

        # horizontal trotter steps
        for x0 in range(NC):
            step_list_h = []
            for i in range(L1):
                for j in range(x0,L2,NC):
    
                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] in ['r','l']:
                            step_list_h.append( [(i,j), opk] )
            if len(step_list_h) > 0:   step_lists.append(['io',step_list_h])

            
        # vertical trotter steps
        for x0 in range(NR):
            step_list_v = []
            for j in range(L2):
                for i in range(x0,L1,NR):

                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] in ['i','o']:
                            step_list_v.append( [(i,j), opk] )
            if len(step_list_v) > 0:   step_lists.append(['lr',step_list_v])


        # other trotter steps   --> just swap L1,L2 to test other env direction
        for (x0,y0) in np.ndindex(NR,NC):
            step_list_x = []
            for i in range(x0,L1,NR):
                for j in range(y0,L2,NC):
    
                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] not in ['r','i','o','l']:
                            step_list_x.append( [(i,j), opk] )
            if len(step_list_x) > 0:   step_lists.append(['io',step_list_x])

        return step_lists
        
    elif version == 1:

        NR, NC = trotterH.it_sh
 
        step_lists = []

        # vertical trotter steps
        for x0 in range(NR):
            step_list_v = []
            for j in range(L2):
                for i in range(x0,L1,NR):

                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] in ['i','o']:
                            step_list_v.append( [(i,j), opk] )
            if len(step_list_v) > 0:   step_lists.append(['lr',step_list_v])


        # horizontal trotter steps
        for x0 in range(NC):
            step_list_h = []
            for i in range(L1):
                for j in range(x0,L2,NC):
    
                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] in ['r','l']:
                            step_list_h.append( [(i,j), opk] )
            if len(step_list_h) > 0:   step_lists.append(['io',step_list_h])


        # other trotter steps   --> just swap L1,L2 to test other env direction
        for (x0,y0) in np.ndindex(NR,NC):
            step_list_x = []
            for i in range(x0,L1,NR):
                for j in range(y0,L2,NC):
    
                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] not in ['r','i','o','l']:
                            step_list_x.append( [(i,j), opk] )
            if len(step_list_x) > 0:   step_lists.append(['io',step_list_x])

        return step_lists
 

    ### apply all trotter steps without specifying even/odd bonds (ie. sequentially)       
    elif version == 2:

        NR, NC = trotterH.it_sh
 
        step_lists = []

        # horizontal trotter steps
        step_list_h = []
        for i in range(L1):
            for j in range(L2):
    
                for opk in trotterH.map[i][j]:
                    if trotterH.conn[opk] in ['r','l']:
                        step_list_h.append( [(i,j), opk] )
        if len(step_list_h) > 0:   step_lists.append(['io',step_list_h])

            
        # vertical trotter steps
        step_list_v = []
        for j in range(L2):
            for i in range(L1):

                for opk in trotterH.map[i][j]:
                    if trotterH.conn[opk] in ['i','o']:
                        step_list_v.append( [(i,j), opk] )
        if len(step_list_v) > 0:   step_lists.append(['lr',step_list_v])


        # other trotter steps   --> just swap L1,L2 to test other env direction
        step_list_x = []
        for i in range(L1):
            for j in range(L2):
    
                for opk in trotterH.map[i][j]:
                    if trotterH.conn[opk] not in ['r','i','o','l']:
                        step_list_x.append( [(i,j), opk] )
        if len(step_list_x) > 0:   step_lists.append(['io',step_list_x])

        return step_lists
        
    elif version == 3:

        NR, NC = trotterH.it_sh
 
        step_lists = []

        # vertical trotter steps
        step_list_v = []
        for j in range(L2):
            for i in range(L1):

                for opk in trotterH.map[i][j]:
                    if trotterH.conn[opk] in ['i','o']:
                        step_list_v.append( [(i,j), opk] )
        if len(step_list_v) > 0:   step_lists.append(['lr',step_list_v])


        # horizontal trotter steps
        step_list_h = []
        for i in range(L1):
            for j in range(L2):
    
                for opk in trotterH.map[i][j]:
                    if trotterH.conn[opk] in ['r','l']:
                        step_list_h.append( [(i,j), opk] )
        if len(step_list_h) > 0:   step_lists.append(['io',step_list_h])

            
        # other trotter steps   --> just swap L1,L2 to test other env direction
        step_list_x = []
        for i in range(L1):
            for j in range(L2):
    
                for opk in trotterH.map[i][j]:
                    if trotterH.conn[opk] not in ['r','i','o','l']:
                        step_list_x.append( [(i,j), opk] )
        if len(step_list_x) > 0:   step_lists.append(['io',step_list_x])

        return step_lists


    ## version 0, but xdir chosen based on lattice dimension
    elif version == 5:  

        # if L1 < L2:    xdir = 'lr'
        # elif L1 >= L2: xdir = 'io'
        # else:          xdir = None

        xdir = 'io'    # always choose L1 >= L2
        # xdir = 'lr'

        NR, NC = trotterH.it_sh
 
        step_lists = []


        # horizontal trotter steps
        for x0 in range(NC):
            step_list_h = []
            for i in range(L1):
                for j in range(x0,L2,NC):
    
                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] in ['r','l']:
                            step_list_h.append( [(i,j), opk] )

            if xdir is None:  xdir_ = 'io' 
            else:             xdir_ = xdir

            if len(step_list_h) > 0:   step_lists.append([xdir_,step_list_h])

            
        # vertical trotter steps
        for x0 in range(NR):
            step_list_v = []
            for i in range(x0,L1,NR):
                for j in range(L2):

                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] in ['i','o']:
                            step_list_v.append( [(i,j), opk] )

            if xdir is None:  xdir_ = 'lr' 
            else:             xdir_ = xdir

            if len(step_list_v) > 0:   step_lists.append([xdir_,step_list_v])


        # other trotter steps   --> just swap L1,L2 to test other env direction
        for (x0,y0) in np.ndindex(NR,NC):
            step_list_x = []
            for i in range(x0,L1,NR):
                for j in range(y0,L2,NC):
    
                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] not in ['r','i','o','l']:
                            step_list_x.append( [(i,j), opk] )

            if xdir is None:  xdir_ = 'io' 
            else:             xdir_ = xdir

            if len(step_list_x) > 0:   step_lists.append([xdir_,step_list_x])

        # print 'step lists', step_lists

        return step_lists


    elif version == 6:  

        # if L1 < L2:    xdir = 'lr'
        # elif L1 >= L2: xdir = 'io'
        # else:          xdir = None

        xdir = 'io'    # always choose L1 >= L2
        # xdir = 'lr'

        NR, NC = trotterH.it_sh
 
        step_lists = []


        # horizontal trotter steps
        for x0 in range(NC-1):
            step_list_h = []
            for i in range(L1):
                for j in range(x0,L2,NC-1):
    
                    for opk in trotterH.map[i][j]:
                        # if trotterH.conn[opk] in ['r','l']:
                        if np.all( x in ['r','l'] for x in trotterH.conn[opk]):
                            step_list_h.append( [(i,j), opk] )

            if xdir is None:  xdir_ = 'io' 
            else:             xdir_ = xdir

            if len(step_list_h) > 0:   step_lists.append([xdir_,step_list_h])

            
        # vertical trotter steps
        for x0 in range(NR-1):
            step_list_v = []
            for i in range(x0,L1,NR-1):
                for j in range(L2):

                    for opk in trotterH.map[i][j]:
                        # if trotterH.conn[opk] in ['i','o']:
                        if np.all( x in ['i','o'] for x in trotterH.conn[opk]):
                            step_list_v.append( [(i,j), opk] )

            if xdir is None:  xdir_ = 'lr' 
            else:             xdir_ = xdir

            if len(step_list_v) > 0:   step_lists.append([xdir_,step_list_v])


        # other trotter steps   --> just swap L1,L2 to test other env direction
        for (x0,y0) in np.ndindex(NR-1,NC-1):
            step_list_x = []
            for i in range(x0,L1,NR-1):
                for j in range(y0,L2,NC-1):
    
                    for opk in trotterH.map[i][j]:
                        if trotterH.conn[opk] not in ['r','i','o','l']:
                            step_list_x.append( [(i,j), opk] )

            if xdir is None:  xdir_ = 'io' 
            else:             xdir_ = xdir

            if len(step_list_x) > 0:   step_lists.append([xdir_,step_list_x])

        # print 'step lists', step_lists

        return step_lists
