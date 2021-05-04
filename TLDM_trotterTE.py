import numpy as np
import time

import tens_fcts as tf
import MPX  
import PEPX
import PEPX_GL
import PEPS
import PEPS_GL
import PEPS_GL_env_nolam as ENV_GL
import TLDM_GL
import TLDM_alsq
import TLDM_GL_env_nolam as ENV_DM
import TLDM_GL_env_nolam_SL as ENV_SL
import Operator_2D as Op
import TimeEvolution as TE
import PEPX_GL_trotterTE as TE_GL


# this is for finite PEPS:  so sweep through all of the lattice sites in a dmrg-like style
# calculate entire environment once, and then just make updates with the sweep


'''
general algorithm:  do calcs with initial psi/env
 - apply horizontal bonds + update environment   -> how to update column environments?
 - apply vertical bonds + update environment     -> how to update row environments?
 - apply diagonal bonds 


'''


def get_exp(trotterH,dt):

    op_keys = trotterH.ops.keys()
    exp_tsteps = {}
    for opk in op_keys:     # trotterOp is in MPO form
         # exp_tsteps[opk] = TE.exact_exp(trotterH.ops[opk], dt)

         # print 'mpo type', [m.dtype for m in trotterH.ops[opk]], dt
         temp_exp = TE.exact_exp(trotterH.ops[opk], dt)
         # print 'block exp', temp_exp.dtype
         ns = len(trotterH.conn[opk])+1
         # exp_tsteps[opk] = MPX.MPX(tf.decompose_block(temp_exp,ns,0,-1,'ijk,...'))
         mpo, s_list = tf.decompose_block(temp_exp,ns,0,-1,'ijk,...',return_s=True)
         mpo_ = PEPX.split_singular_vals(mpo,None,s_list)
         # exp_tsteps[opk] = temp_exp
         exp_tsteps[opk] = MPX.MPX(mpo_)
         # print 'mpo decomposition err', np.linalg.norm((MPX.MPX(mpo_)).getSites()-temp_exp)
         # print 'exp type', [m.dtype for m in mpo]

    return exp_tsteps



##################################
#### update sites via SU or FU ###
##################################

def update_sites_SU(tldm,ind0,op_conn,op,DMAX=10):

    gam, lam, axT_invs = PEPX_GL.get_sites(tldm,ind0,op_conn)
    gam_, lam_, errs = PEPS_GL.simple_update(gam, lam, op, DMAX=DMAX, direction=0, normalize=True)
    # gam_, lam_ = PEPX_GL.regularize_GL_list(gam_,lam_)

    # print 'update SU'
    # PEPX_GL.check_GL_canonical(gam_,lam_)
    # print 'done update SU check'

    tldm_ = PEPX_GL.set_sites(tldm,ind0,op_conn,gam_,lam_,axT_invs)

    # gam, lam, axT_invs = PEPX_GL.get_sites(tldm_,ind0,op_conn)
    # print 'update TLDM'
    # PEPX_GL.check_GL_canonical(gam,lam)
    # print 'done update TLDM check'

    return TLDM_GL.TLDM_GL(tldm_)


def update_sites_FU(tldm,ind0,op_conn,op,envs_list,DMAX=10,XMAX=100,build_env=False,qr_reduce=False,
                    ensure_pos=False, alsq_2site=False):

    xs,ys = PEPX.get_conn_inds(op_conn,ind0)

    tldm_ = PEPS_GL.alsq_block_update(tldm,op_conn,[xs,ys],envs_list,op,DMAX=DMAX,XMAX=XMAX,site_idx=None,
                                      normalize=False,build_env=build_env,flatten=True,qr_reduce=qr_reduce,
                                      ensure_pos=ensure_pos,alsq_2site=alsq_2site) #,init_guess=tldm)
    # tldm_ = TLDM_alsq.alsq_block_update(tldm,op_conn,[xs,ys],envs_list,op,DMAX=DMAX,XMAX=XMAX,site_idx=None,
    #                                     normalize=False,build_env=build_env,qr_reduce=qr_reduce,
    #                                     ensure_pos=ensure_pos) #,init_guess=tldm)
    return tldm_
    


##################################
#### SIMPLE UPDATE TROTTER    ####
##################################

# @profile
def apply_trotter_layer_SU(tldm, step_list, trotterH, expH=None, DMAX=10):

    tldm_ = tldm.copy()
    L1,L2 = tldm.shape

    for ind, opk in step_list:

        # print 'SU step init', ind, opk
        # if L1 > 1 and L2 == 1:
        #     gs,ls,axT = PEPX_GL.get_sites( tldm_, (0,0), 'o'*(L1-1))
        #     gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls)
        #     # print 'norm',TLDM_GL.norm(tldm)
        #     PEPX_GL.check_GL_canonical(gs_,ls_)
        #     tldm_ = PEPX_GL.set_sites( tldm_, (0,0), 'o'*(L1-1), gs_, ls_, axT)

        # if L1 == 1 and L2 > 1:
        #     gs,ls,axT = PEPX_GL.get_sites( tldm_, (0,0), 'r'*(L2-1))
        #     gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls)
        #     # print 'norm',TLDM_GL.norm(tldm)
        #     PEPX_GL.check_GL_canonical(gs_,ls_)
        #     tldm_ = PEPX_GL.set_sites( tldm_, (0,0), 'r'*(L2-1), gs_, ls_, axT)

        i,j = ind

        NR, NC  = trotterH.ns[opk]
        op_conn = trotterH.conn[opk]
        ind0    = trotterH.ind0[opk]

        if expH is None:      op  = trotterH.ops[opk]
        else:                 op  = expH[opk]

        sub_tldm = tldm_[i:i+NR,j:j+NC]
        sub_tldm = update_sites_SU(sub_tldm,ind0,op_conn,op,DMAX=DMAX)
        
        tldm_[i:i+NR,j:j+NC] = sub_tldm
        # print i,j, NR, NC

        # print 'trotter SU step norm', TLDM_GL.norm(tldm_)

        # print 'check trotter step'
        if L1 > 1 and L2 == 1: # and NR==2:
            gs,ls,axT = PEPX_GL.get_sites( tldm_, (0,0), 'o'*(L1-1))
            PEPX_GL.check_GL_canonical(gs[:i+NR-1],ls[:i+NR], check='L')
            PEPX_GL.check_GL_canonical(gs[i+NR-1:],ls[i+NR-1:], check='R')
        # print 'end check trotter step'

    return tldm_



#####################################
#### DOUBLE LAYER CONTRACTION FU ####
#####################################

### flatten in alsq loop so that can use same exp(iHt) operators
def apply_trotter_layer_IO(tldm, step_list, trotterH, expH=None, DMAX=10, XMAX=100, build_env=False,
                           qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using I and O as main boundaries'''

    tldm_ = tldm.copy()

    envIs = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'I', 0, XMAX=XMAX)  # list of len ii+1
    envOs = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'O', 0, XMAX=XMAX)  # list of len L+1


    # print 'init norm tldm', TLDM_GL.norm(tldm)

    i_, j_ = 0,0

    for ind, opk in step_list:    # assumes that these inds are order top to bottom, left to right

        i,j = ind

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
                b_mpo,err = ENV_DM.get_next_boundary_I(np.conj(tldm_[ix,:]),tldm_[ix,:],envIs[ix],XMAX=XMAX)
                envIs.append(b_mpo)

            # temp_mpo, err = ENV_DM.get_next_boundary_I(np.conj(tldm_[i,:]),tldm_[i,:],envIs[i],XMAX=XMAX)
            # temp_ovlp = ENV_DM.contract_2_bounds(temp_mpo,envOs[i+1],
            #                                  tldm_.lambdas[i+1,:,1],tldm_.lambdas[i+1,:,1])
            # print 'new boundI --> norm', i, len(envIs), temp_ovlp

            envI = envIs[i]
            envO = envOs[i+NR]

            senvRs = ENV_DM.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'R',0,XMAX=XMAX)
                     # list of len L+1
            senvLs = ENV_DM.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'L',j,XMAX=XMAX)
                     # list of len jj+2

        try:
            envL = senvLs[j]
            envR = senvRs[j+NC]
        except(UnboundLocalError):
            # print 'unbound error'

            senvRs = ENV_DM.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'R',0,XMAX=XMAX)
                     # list of len L+1
            senvLs = ENV_DM.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'L',j,XMAX=XMAX)
                     # list of len jj+2
            envL = senvLs[j]
            envR = senvRs[j+NC]

        except(IndexError):

            # print 'index error', i, j

            for jx in range(j_,j):
                e_mpo, err = ENV_DM.get_next_subboundary_L(senvLs[jx],envIs[i][jx],np.conj(tldm_[i:i+NR,jx]),
                                                           tldm_[i:i+NR,jx],envOs[i+NR][jx],XMAX=XMAX)
                senvLs.append(e_mpo)
            envL = senvLs[j]
            envR = senvRs[j+NC]


        # # env without lambda on dangling bonds
        # lam_i = [tldm_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        # bi = ENV_DM.apply_lam_to_boundary_mpo(envI[j:j+NC], lam_i, lam_i)
        # lam_o = [tldm_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        # bo = ENV_DM.apply_lam_to_boundary_mpo(envO[j:j+NC], lam_o, lam_o)
        # lam_l = [tldm_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        # bl = ENV_DM.apply_lam_to_boundary_mpo(envL, lam_l, lam_l)
        # lam_r = [tldm_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        # br = ENV_DM.apply_lam_to_boundary_mpo(envR, lam_r, lam_r)

        bi = envI[j:j+NC]
        bo = envO[j:j+NC]
        bl = envL
        br = envR

        sub_tldm = tldm_[i:i+NR,j:j+NC]

        # print 'step opk', i,j, opk

        # print 'init norm xx', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'init norm xx', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)
       

        sub_tldm = update_sites_FU(sub_tldm,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        # print 'IO DL updated lambdas', i,j,[sub_tldm.lambdas[idx] for idx in np.ndindex(sub_tldm.shape)]
        # print type(sub_tldm)
        tldm_[i:i+NR,j:j+NC] = sub_tldm

        # print 'norm', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'tldm norm', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)
        # # exit()
        
        i_,j_ = i,j


    for ix in range(i,i+NR):
        b_mpo,err = ENV_DM.get_next_boundary_I(np.conj(tldm_[ix,:]),tldm_[ix,:],envIs[ix],XMAX=XMAX)
        envIs.append(b_mpo)

    ovlp = ENV_DM.contract_2_bounds( envIs[i+NR], envOs[i+NR], tldm_.lambdas[i+NR-1,:,2], tldm_.lambdas[i+NR-1,:,2] )
    norm = np.sqrt( ovlp )

    return tldm_, norm


# @profile
def apply_trotter_layer_LR(tldm, step_list, trotterH, expH=None, DMAX=10, XMAX=100, build_env=False,
                           qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using L and R as main boundaries'''

    # print 'LR layer'

    tldm_ = tldm.copy()

    envLs = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'L', 0, XMAX=XMAX)  # list of len ii+1
    envRs = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'R', 0, XMAX=XMAX)  # list of len L+1

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
                b_mpo,err = ENV_DM.get_next_boundary_L(np.conj(tldm_[:,jx]),tldm_[:,jx],envLs[jx],XMAX=XMAX)
                envLs.append(b_mpo)
            envL = envLs[j]
            envR = envRs[j+NC]

            senvOs = ENV_DM.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'O',0,XMAX=XMAX)
                     # list of len L+1
            senvIs = ENV_DM.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'I',i,XMAX=XMAX)
                     # list of len jj+2

        try:
            envI = senvIs[i]
            envO = senvOs[i+NR]

        except(UnboundLocalError):
            senvOs = ENV_DM.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'O',0,XMAX=XMAX)
                     # list of len L+1
            senvIs = ENV_DM.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'I',i,XMAX=XMAX)
                     # list of len jj+2
            envI = senvIs[i]
            envO = senvOs[i+NR]

        except(IndexError):
            for ix in range(i_,i):
                e_mpo, err = ENV_DM.get_next_subboundary_I(senvIs[ix],envLs[j][ix],np.conj(tldm_[ix,j:j+NC]),
                                                           tldm_[ix,j:j+NC],envRs[j+NC][ix],XMAX=XMAX)
                senvIs.append(e_mpo)
            envI = senvIs[i]
            envO = senvOs[i+NR]


        # # env without lambda on dangling bonds
        # lam_i = [tldm_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        # bi = ENV_DM.apply_lam_to_boundary_mpo(envI, lam_i, lam_i)
        # lam_o = [tldm_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        # bo = ENV_DM.apply_lam_to_boundary_mpo(envO, lam_o, lam_o)
        # lam_l = [tldm_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        # bl = ENV_DM.apply_lam_to_boundary_mpo(envL[i:i+NR], lam_l, lam_l)
        # lam_r = [tldm_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        # br = ENV_DM.apply_lam_to_boundary_mpo(envR[i:i+NR], lam_r, lam_r)

        bi = envI
        bo = envO
        bl = envL[i:i+NR]
        br = envR[i:i+NR]

        sub_tldm = tldm_[i:i+NR,j:j+NC]

        # print 'step opk', i,j, opk

        # print 'init norm xx', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'init norm xx', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)
       
        sub_tldm = update_sites_FU(sub_tldm,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        # print 'LR DL updated lambdas', i,j,[sub_tldm.lambdas[idx] for idx in np.ndindex(sub_tldm.shape)]
        tldm_[i:i+NR,j:j+NC] = sub_tldm

        # print 'norm', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'tldm norm', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)
        # # exit()

        i_,j_ = i,j


    for jx in range(j,j+NC):
        b_mpo,err = ENV_DM.get_next_boundary_L(np.conj(tldm_[:,jx]),tldm_[:,jx],envLs[jx],XMAX=XMAX)
        envLs.append(b_mpo)

    ovlp = ENV_DM.contract_2_bounds( envLs[j+NC], envRs[j+NC], tldm_.lambdas[:,j+NC-1,3], tldm_.lambdas[:,j+NC-1,3] )
    norm = np.sqrt( ovlp )

    return tldm_, norm


#####################################
#### SINGLE LAYER CONTRACTION FU ####
#####################################

def apply_trotter_layer_IO_SL(tldm, step_list, trotterH, expH=None, DMAX=10, XMAX=100, scaleX=1, build_env=False,
                              qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using I and O as main boundaries'''

    tldm_ = tldm.copy()

    envIs = ENV_SL.get_boundaries( np.conj(tldm_), tldm_, 'I', 0, XMAX=XMAX, scaleX=scaleX)  # list of len ii+1
    envOs = ENV_SL.get_boundaries( np.conj(tldm_), tldm_, 'O', 0, XMAX=XMAX, scaleX=scaleX)  # list of len L+1

    # #############
    # envIsD = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'I', 0, XMAX=XMAX)  # list of len ii+1
    # envOsD = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'O', 0, XMAX=XMAX)  # list of len L+1


    i_, j_ = 0,0

    # print 'IO SL fct'

    for ind, opk in step_list:    # assumes that these inds are order top to bottom, left to right

        i,j = ind

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
                b_mpo,err = ENV_SL.get_next_boundary_I(np.conj(tldm_[ix,:]),tldm_[ix,:],envIs[ix],XMAX=XMAX,
                                                       scaleX=scaleX)
                envIs.append(b_mpo)

            envI = envIs[i]
            envO = envOs[i+NR]

            senvRs = ENV_SL.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'R',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvLs = ENV_SL.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'L',j,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len jj+2

            # #############
            # for ix in range(i_,i):
            #     b_mpo,err = ENV_DM.get_next_boundary_I(np.conj(tldm_[ix,:]),tldm_[ix,:],envIsD[ix],XMAX=XMAX)
            #     envIsD.append(b_mpo)
            # envID = envIsD[i]
            # envOD = envOsD[i+NR]

            # senvRsD = ENV_DM.get_subboundaries(envID,envOD,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'R',0,XMAX=XMAX)
            #          # list of len L+1
            # senvLsD = ENV_DM.get_subboundaries(envID,envOD,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'L',j,XMAX=XMAX)
            #          # list of len jj+2

        try:
            envL = senvLs[j]
            envR = senvRs[j+NC]

            # #############
            # envLD = senvLsD[j]
            # envRD = senvRsD[j+NC]

        except(UnboundLocalError):
            # print 'unbound error'

            senvRs = ENV_SL.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'R',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvLs = ENV_SL.get_subboundaries(envI,envO,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'L',j,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len jj+2
            envL = senvLs[j]
            envR = senvRs[j+NC]

            # #############
            # senvRsD = ENV_DM.get_subboundaries(envID,envOD,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'R',0,XMAX=XMAX)
            #          # list of len L+1
            # senvLsD = ENV_DM.get_subboundaries(envID,envOD,np.conj(tldm_[i:i+NR,:]),tldm_[i:i+NR,:],'L',j,XMAX=XMAX)
            #          # list of len jj+2
            # envLD = senvLsD[j]
            # envRD = senvRsD[j+NC]

        except(IndexError):

            # print 'index error', i, j

            for jx in range(j_,j):
                jx_ = 2*jx
                e_mpo, err = ENV_SL.get_next_subboundary_L(senvLs[jx],envIs[i][jx_:jx_+2],
                                                           np.conj(tldm_[i:i+NR,jx]),tldm_[i:i+NR,jx],
                                                           envOs[i+NR][jx_:jx_+2],XMAX=XMAX,scaleX=scaleX)
                senvLs.append(e_mpo)
            envL = senvLs[j]
            envR = senvRs[j+NC]

            # #############
            # for jx in range(j_,j):
            #     e_mpo, err = ENV_DM.get_next_subboundary_L(senvLsD[jx],envIsD[i][jx],
            #                                                np.conj(tldm_[i:i+NR,jx]),tldm_[i:i+NR,jx],
            #                                                envOsD[i+NR][jx],XMAX=XMAX)
            #     senvLsD.append(e_mpo)
            # envLD = senvLsD[j]
            # envRD = senvRsD[j+NC]


        envI_DL = ENV_SL.SL_to_DL_bound(envI,'row')
        envO_DL = ENV_SL.SL_to_DL_bound(envO,'row')
        envL_DL = ENV_SL.SL_to_DL_bound(envL,'col')
        envR_DL = ENV_SL.SL_to_DL_bound(envR,'col')

        # # env without lambda on dangling bonds
        # lam_i = [tldm_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        # bi = ENV_DM.apply_lam_to_boundary_mpo(envI_DL[j:j+NC], lam_i, lam_i)
        # lam_o = [tldm_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        # bo = ENV_DM.apply_lam_to_boundary_mpo(envO_DL[j:j+NC], lam_o, lam_o)
        # lam_l = [tldm_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        # bl = ENV_DM.apply_lam_to_boundary_mpo(envL_DL, lam_l, lam_l)
        # lam_r = [tldm_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        # br = ENV_DM.apply_lam_to_boundary_mpo(envR_DL, lam_r, lam_r)

        bi = envI_DL[j:j+NC]
        bo = envO_DL[j:j+NC]
        bl = envL_DL
        br = envR_DL

        # make_env_hermitian = False
        # if make_env_hermitian:
        #     print 'env hermitian'
        #     bi = ENV_GL.make_env_hermitian(bi,DMAX=XMAX,direction=0,use_tens=True)
        #     bo = ENV_GL.make_env_hermitian(bo,DMAX=XMAX,direction=1,use_tens=True)
        #     bl = ENV_GL.make_env_hermitian(bl,DMAX=XMAX,direction=1,use_tens=True)
        #     br = ENV_GL.make_env_hermitian(br,DMAX=XMAX,direction=0,use_tens=True)

        # print 'step opk', i,j, opk

        # #############
        # env_SL = ENV_GL.build_env([bl,bi,bo,br])

        # biD = envID[j:j+NC]
        # boD = envOD[j:j+NC]
        # blD = envLD
        # brD = envRD
        # env_DL = ENV_GL.build_env([blD,biD,boD,brD])
        # env_err = np.linalg.norm(env_SL-env_DL)
        # env_norm = np.linalg.norm(env_SL)
        # # if env_err > 1.0e-5:
        # #     print 'warning: DL vs SL IO env difference is large', env_err
        # if env_err/env_norm > 1.0e-1:
        #     print 'IO env err', i,j,env_err,env_norm, np.linalg.norm(env_DL)
        #     print [np.linalg.norm(m) for idx,m in np.ndenumerate(tldm_)]
        #     # print [tldm_.lambdas[idx] for idx in np.ndindex(tldm_.shape)]
        #     # print PEPX_GL.norm(tldm_,XMAX=XMAX)
        #     # print [np.linalg.norm(m) for m in biD], [np.linalg.norm(m) for m in bi]
        #     # print [np.linalg.norm(m) for m in boD], [np.linalg.norm(m) for m in bo]
        #     # print [np.linalg.norm(m) for m in blD], [np.linalg.norm(m) for m in bl]
        #     # print [np.linalg.norm(m) for m in brD], [np.linalg.norm(m) for m in br]
        #     # exit()
 

        ### other test ###
        # full_env = ENV_GL.build_env([bl,bi,bo,br])
        # print 'env norm', np.linalg.norm(full_env)
        # full_env_sq = np.transpose( full_env, [m for m in range(0,full_env.ndim,2)] +\
        #                                       [m for m in range(1,full_env.ndim,2)] )
        # sqdim = int(np.sqrt(np.prod(full_env_sq.shape)))
        # full_env_sq = full_env_sq.reshape(sqdim,sqdim)
        # print 'env - I', np.linalg.norm(full_env_sq - np.eye(sqdim))
        # evals = np.linalg.eigvals(full_env_sq)
        # print 'neg evals',np.linalg.norm( evals[ np.where(evals < 0)[0] ])
        

        sub_tldm = tldm_[i:i+NR,j:j+NC]

        # print 'init norm xx', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'temp', [m.shape for idx,m in np.ndenumerate(temp)], [m.shape for m in bl],
        # print [m.shape for m in bi],[m.shape for m in bo],[m.shape for m in br]
        # print 'init norm xx', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)
       
        sub_tldm = update_sites_FU(sub_tldm,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        # print 'IO SL updated lambdas', i,j,[sub_tldm.lambdas[idx] for idx in np.ndindex(sub_tldm.shape)]
        tldm_[i:i+NR,j:j+NC] = sub_tldm

        # print 'norm', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'tldm norm', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)

        # exit()
        # print 'updated norm', ENV_DM.embed_sites_norm( sub_tldm, [bl,bi,bo,br] )

        i_,j_ = i,j



    for ix in range(i,i+NR):
        b_mpo,err = ENV_SL.get_next_boundary_I(np.conj(tldm_[ix,:]),tldm_[ix,:],envIs[ix],XMAX=XMAX,scaleX=scaleX)
        envIs.append(b_mpo)

    ovlp = ENV_SL.contract_2_bounds(envIs[i+NR],envOs[i+NR],tldm_.lambdas[i+NR-1,:,2],tldm_.lambdas[i+NR-1,:,2],'row')
    norm = np.sqrt( ovlp )

    return tldm_, norm


def apply_trotter_layer_LR_SL(tldm, step_list, trotterH, expH=None, DMAX=10, XMAX=100, scaleX=1, build_env=False,
                           qr_reduce=False, ensure_pos=False, alsq_2site=False):
    ''' perform trotter step using L and R as main boundaries'''

    tldm_ = tldm.copy()

    envLs = ENV_SL.get_boundaries( np.conj(tldm_), tldm_, 'L', 0, XMAX=XMAX,scaleX=scaleX)  # list of len ii+1
    envRs = ENV_SL.get_boundaries( np.conj(tldm_), tldm_, 'R', 0, XMAX=XMAX,scaleX=scaleX)  # list of len L+1

    # #############
    # envLsD = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'L', 0, XMAX=XMAX)  # list of len ii+1
    # envRsD = ENV_DM.get_boundaries( np.conj(tldm_), tldm_, 'R', 0, XMAX=XMAX)  # list of len L+1

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
                b_mpo,err = ENV_SL.get_next_boundary_L(np.conj(tldm_[:,jx]),tldm_[:,jx],envLs[jx],
                                                       XMAX=XMAX,scaleX=scaleX)
                envLs.append(b_mpo)
            envL = envLs[j]
            envR = envRs[j+NC]

            senvOs = ENV_SL.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'O',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvIs = ENV_SL.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'I',i,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len jj+2

            # #############
            # for jx in range(j_,j):
            #     b_mpo,err = ENV_GL.get_next_boundary_L(np.conj(tldm_[:,jx]),tldm_[:,jx],envLsD[jx],XMAX=XMAX)
            #     envLsD.append(b_mpo)
            # envLD = envLsD[j]
            # envRD = envRsD[j+NC]

            # senvOsD = ENV_DM.get_subboundaries(envLD,envRD,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'O',0,XMAX=XMAX)
            #          # list of len L+1
            # senvIsD = ENV_DM.get_subboundaries(envLD,envRD,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'I',i,XMAX=XMAX)
            #          # list of len jj+2

        try:
            envI = senvIs[i]
            envO = senvOs[i+NR]

            # #############
            # envID = senvIsD[i]
            # envOD = senvOsD[i+NR]

        except(UnboundLocalError):
            senvOs = ENV_SL.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'O',0,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len L+1
            senvIs = ENV_SL.get_subboundaries(envL,envR,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'I',i,
                                              XMAX=XMAX,scaleX=scaleX)
                     # list of len jj+2
            envI = senvIs[i]
            envO = senvOs[i+NR]

            # #############
            # senvOsD = ENV_DM.get_subboundaries(envLD,envRD,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'O',0,XMAX=XMAX)
            #          # list of len L+1
            # senvIsD = ENV_DM.get_subboundaries(envLD,envRD,np.conj(tldm_[:,j:j+NC]),tldm_[:,j:j+NC],'I',i,XMAX=XMAX)
            #          # list of len jj+2
            # envID = senvIsD[i]
            # envOD = senvOsD[i+NR]

        except(IndexError):
            for ix in range(i_,i):
                ix_ = 2*ix
                e_mpo, err = ENV_SL.get_next_subboundary_I(senvIs[ix],envLs[j][ix_:ix_+2],
                                                           np.conj(tldm_[ix,j:j+NC]),tldm_[ix,j:j+NC],
                                                           envRs[j+NC][ix_:ix_+2],XMAX=XMAX,scaleX=scaleX)
                senvIs.append(e_mpo)
            envI = senvIs[i]
            envO = senvOs[i+NR]

            # #############
            # for ix in range(i_,i):
            #     e_mpo, err = ENV_DM.get_next_subboundary_I(senvIsD[ix],envLsD[j][ix],
            #                                                np.conj(tldm_[ix,j:j+NC]),tldm_[ix,j:j+NC],
            #                                                envRsD[j+NC][ix],XMAX=XMAX)
            #     senvIsD.append(e_mpo)
            # envID = senvIsD[i]
            # envOD = senvOsD[i+NR]


        envI_DL = ENV_SL.SL_to_DL_bound(envI,'row')
        envO_DL = ENV_SL.SL_to_DL_bound(envO,'row')
        envL_DL = ENV_SL.SL_to_DL_bound(envL,'col')
        envR_DL = ENV_SL.SL_to_DL_bound(envR,'col')

        # # env without lambda on dangling bonds
        # lam_i = [tldm_.lambdas[i,jx,1] for jx in range(j,j+NC)]
        # bi = ENV_DM.apply_lam_to_boundary_mpo(envI_DL, lam_i, lam_i)
        # lam_o = [tldm_.lambdas[i+NR-1,jx,2] for jx in range(j,j+NC)]
        # bo = ENV_DM.apply_lam_to_boundary_mpo(envO_DL, lam_o, lam_o)
        # lam_l = [tldm_.lambdas[ix,j,0] for ix in range(i,i+NR)]
        # bl = ENV_DM.apply_lam_to_boundary_mpo(envL_DL[i:i+NR], lam_l, lam_l)
        # lam_r = [tldm_.lambdas[ix,j+NC-1,3] for ix in range(i,i+NR)]
        # br = ENV_DM.apply_lam_to_boundary_mpo(envR_DL[i:i+NR], lam_r, lam_r)

        bi = envI_DL
        bo = envO_DL
        bl = envL_DL[i:i+NR]
        br = envR_DL[i:i+NR]

        # make_env_hermitian = False
        # if make_env_hermitian:
        #     print 'env hermitian'
        #     bi = ENV_GL.make_env_hermitian(bi,DMAX=XMAX,direction=0,use_tens=True)
        #     bo = ENV_GL.make_env_hermitian(bo,DMAX=XMAX,direction=1,use_tens=True)
        #     bl = ENV_GL.make_env_hermitian(bl,DMAX=XMAX,direction=1,use_tens=True)
        #     br = ENV_GL.make_env_hermitian(br,DMAX=XMAX,direction=0,use_tens=True)

        # print 'step opk', i,j, opk

        # #############
        # env_SL = ENV_GL.build_env([bl,bi,bo,br])

        # biD = envID
        # boD = envOD
        # blD = envLD[i:i+NR]
        # brD = envRD[i:i+NR]
        # env_DL = ENV_GL.build_env([blD,biD,boD,brD])
        # env_err = np.linalg.norm( env_DL - env_SL )
        # env_norm = np.linalg.norm(env_SL)
        # # if env_err > 1.0e-5:
        # #     print 'warning:  DL vs SL LR env norm diff bad', env_err
        # if env_err/env_norm > 1.0e-1:
        #     print 'LR env err', i,j, env_err, env_norm, np.linalg.norm(env_DL)
        #     print [np.linalg.norm(m) for idx,m in np.ndenumerate(tldm_)]
        #     # print [tldm_.lambdas[idx] for idx in np.ndindex(tldm_.shape)]
        #     # print PEPX_GL.norm(tldm_,XMAX=XMAX)
        #     # print [np.linalg.norm(m) for m in biD], [np.linalg.norm(m) for m in bi]
        #     # print [np.linalg.norm(m) for m in boD], [np.linalg.norm(m) for m in bo]
        #     # print [np.linalg.norm(m) for m in blD], [np.linalg.norm(m) for m in bl]
        #     # print [np.linalg.norm(m) for m in brD], [np.linalg.norm(m) for m in br]
        #     # exit()

        sub_tldm = tldm_[i:i+NR,j:j+NC]

        ### other test ###

        # print 'init norm xx', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'init norm xx', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)
       
        sub_tldm = update_sites_FU(sub_tldm,ind0,op_conn,op,[bl,bi,bo,br],DMAX,XMAX,build_env,qr_reduce,
                                   ensure_pos,alsq_2site)
        # print 'LR SL updated lambdas', i,j,[sub_tldm.lambdas[idx] for idx in np.ndindex(sub_tldm.shape)]
        tldm_[i:i+NR,j:j+NC] = sub_tldm

        # print 'norm', ENV_DM.embed_sites_norm(sub_tldm, [bl,bi,bo,br]) #, TLDM_GL.norm(tldm_)
        # temp = PEPX_GL.flatten(sub_tldm)
        # print 'tldm norm', ENV_GL.embed_sites_norm(temp, [bl,bi,bo,br]) #, PEPX_GL.norm(temp)

        i_,j_ = i,j


    for jx in range(j,j+NC):
        b_mpo,err = ENV_SL.get_next_boundary_L(np.conj(tldm_[:,jx]),tldm_[:,jx],envLs[jx],XMAX=XMAX,scaleX=scaleX)
        envLs.append(b_mpo)

    ovlp = ENV_SL.contract_2_bounds(envLs[j+NC],envRs[j+NC],tldm_.lambdas[:,j+NC-1,3],tldm_.lambdas[:,j+NC-1,3],'col')
    norm = np.sqrt( ovlp )

    return tldm_, norm


###############################
### perfrom trotter step TE ###
###############################

# @profile
def run_TE(dt,totT,trotterH,initTLDM,obs_pepos={},DMAX=10,XMAX=100,XMAX2=100,normalize=True,
           te_type='SU',build_env=False,qr_reduce=False,ensure_pos=False,trotter_version=2,contract_SL=True,
           scaleX=1,truncate_run=False,alsq_2site=False,run_tol=1.0e-6):
    ''' time evolution algorithm
        dt, totT = time step, total amounth of time
        trotterH = [list of trotter steps generating H, corresponding orientations 'h','v','hv']
                   len(orientation) matches len(trotter mpo).   eg. 'hv' = diagonal, 'hh' = horizontal over 2 sites
        trotterH = dictionary of trotter steps
                   key = orientation 'h', 'v', 'hv', 
                   ... length of key = len(trotter mpo), 'hv' = diagonal, 'hh' = horizontal over 2 sites
                   mapped to MPO
        initTLDM = initial thermal PEPO
    '''

    L1,L2 = initTLDM.shape
      
    exp_tsteps = get_exp(trotterH,dt)

    # if obs_pepo is None:  obs_pepo = Op.PEPO_mag((L1,L2),'SZ').getPEPO()


    # start time evolution via trotter stpes
    tldm = initTLDM.copy()
    dbs  = tldm.phys_bonds.copy()

    tstep = 0
    numSteps = int(round(abs(totT/dt),2))

    obs_t = {}
    for obs_key in obs_pepos.keys():
        obs_t[obs_key] = []


    step_lists = get_trotter_layers(trotterH,version=trotter_version)

    while tstep < numSteps:

        for xdir, step_list in step_lists:

            if te_type in ['FU','full']:
                if contract_SL:
                    if xdir in ['io','oi']:
                        tldm, norm = apply_trotter_layer_IO_SL(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,scaleX,
                                                        build_env,qr_reduce,ensure_pos,alsq_2site)
                    elif xdir in ['lr','rl']:
                        tldm, norm = apply_trotter_layer_LR_SL(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,scaleX,
                                                        build_env,qr_reduce,ensure_pos,alsq_2site)
                else:
                    if xdir in ['io','oi']:
                        tldm, norm = apply_trotter_layer_IO(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                        build_env,qr_reduce,ensure_pos,alsq_2site)
                    elif xdir in ['lr','rl']:
                        tldm, norm = apply_trotter_layer_LR(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                        build_env,qr_reduce,ensure_pos,alsq_2site)

                tldm = TLDM_GL.mul(1./norm, tldm)
                # print 'fu norm', norm
                # print [tldm.lambdas[idx] for idx in np.ndindex(tldm.shape)]

            elif te_type in ['SU','simple']:
                tldm = apply_trotter_layer_SU(tldm,step_list,trotterH,exp_tsteps,DMAX)

                # normalize_step = False
                # if normalize_step:
                #     norm = TLDM_GL.norm(tldm,XMAX=XMAX)
                #     tldm = PEPX_GL.mul(1./norm, tldm)


        # # need this for 2D where things don't really stay canonical
        # if te_type in ['SU','simple','FU']:
        #     norm = TLDM_GL.norm(tldm,XMAX=XMAX2)
        #     tldm = TLDM_GL.mul(1./norm,tldm)

        ## measure observables
        for obs_key in obs_pepos.keys():
            obs,norm = TLDM_GL.meas_obs(tldm,obs_pepos[obs_key],XMAX=XMAX2,return_norm=True,contract_SL=contract_SL)
            tldm = TLDM_GL.mul(1./norm,tldm)  ## obs is already normalized

            # obs = TLDM_GL.meas_obs(tldm,obs_pepos[obs_key],XMAX=XMAX2,return_norm=False,
            #                              contract_SL=contract_SL,scaleX=scaleX)
            obs_t[obs_key] += [obs]

        try:
            obs = obs_t['H'][-1]
            obs_key = 'H'
            H_diff = obs - np.min(obs_t['H'])
        except:
            H_diff = -1

        try:
            if isinstance(obs,np.float) or isinstance(obs,np.complex):    print 'obs', tstep, obs_key, obs/L1/L2
        except(UnboundLocalError):
            print 'obs', tstep    # no obs defined

        # if truncate_run and H_diff > 1.0e-6:
        if truncate_run and (H_diff > 1.0e-6 or np.abs(H_diff) < run_tol):
            print 'energy went up or within run tolerance; truncating TE run', H_diff
            break

        # print 'norm', norm
        # tldm = TLDM_GL.mul(1./norm, tldm)


        ## ensure canonicalization if possible
        # print 'check time step'
        if L1 > 1 and L2 == 1:
            gs,ls,axT = PEPX_GL.get_sites( tldm, (0,0), 'o'*(L1-1))
            # PEPX_GL.check_GL_canonical(gs,ls)

            gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=0)
            # print 'check canon'
            PEPX_GL.check_GL_canonical(gs_,ls_)
            tldm = PEPX_GL.set_sites( tldm, (0,0), 'o'*(L1-1), gs_, ls_, axT)

        if L1 == 1 and L2 > 1:
            gs,ls,axT = PEPX_GL.get_sites( tldm, (0,0), 'r'*(L2-1))
            # PEPX_GL.check_GL_canonical(gs,ls)

            gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=1)
            # print 'norm',TLDM_GL.norm(tldm)
            PEPX_GL.check_GL_canonical(gs_,ls_)
            tldm = PEPX_GL.set_sites( tldm, (0,0), 'r'*(L2-1), gs_, ls_, axT)
        # print 'end check time step'

        
        # if tstep == 1:  exit()

        # exit()

        tstep += 1

    return obs_t, tldm


def run_TE_listH(dt,totT,trotterHs,initTLDM,obs_pepos={},DMAXs=[10],XMAXs=[100],XMAX2s=[100],te_types=['SU'],
                 use_exps=[True], normalize=True,build_env=False,qr_reduce=False,ensure_pos=False,trotter_version=2,
                 contract_SL=True, truncate_run=False,run_tol=1.0e-6):

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

    L1,L2 = initTLDM.shape
    tldm = initTLDM.copy()
    dbs  = tldm.phys_bonds.copy()
    
    time1 = time.time()


    obs_t = {}
    for obs_key in obs_pepos.keys():
        obs_t[obs_key] = []

    exp_tsteps_list = []
    for i in range(len(trotterHs)):
        if use_exps[i%len(use_exps)]:     exp_tsteps_list += [ get_exp(trotterHs[i],dt) ]
        else:                             exp_tsteps_list += [None]

    step_lists_list = [ get_trotter_layers(trotterH,version=trotter_version) for trotterH in trotterHs ]


    # start time evolution via trotter stpes
    tstep = 0
    numSteps = int(abs(totT/dt))
    while tstep < numSteps:

        step_ind = 0 
        for step_lists in step_lists_list:

            DMAX  = DMAXs [step_ind%len(DMAXs)]
            XMAX  = XMAXs [step_ind%len(XMAXs)]
            XMAX2 = XMAX2s[step_ind%len(XMAX2s)]
            te_type = te_types[step_ind%len(te_types)]

            exp_tsteps = exp_tsteps_list[step_ind]
            trotterH = trotterHs[step_ind]
            # print 'trotter', step_ind

            for xdir, step_list in step_lists:

                if te_type in ['FU','full']:
                    if contract_SL:
                        if xdir in ['io','oi']:
                            tldm, norm = apply_trotter_layer_IO_SL(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 qr_reduce=qr_reduce,ensure_pos=ensure_pos)
                        elif xdir in ['lr','rl']:
                            tldm, norm = apply_trotter_layer_LR_SL(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 qr_reduce=qr_reduce,ensure_pos=ensure_pos)
                    else:
                        if xdir in ['io','oi']:
                            tldm, norm = apply_trotter_layer_IO(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 build_env,qr_reduce,ensure_pos)
                        elif xdir in ['lr','rl']:
                            tldm, norm = apply_trotter_layer_LR(tldm,step_list,trotterH,exp_tsteps,DMAX,XMAX,
                                                                 build_env,qr_reduce,ensure_pos)

                    tldm = TLDM_GL.mul(1./norm, tldm)
                    # print 'fu norm', norm

                elif te_type in ['SU','simple']:

                    tldm = apply_trotter_layer_SU(tldm,step_list,trotterH,exp_tsteps,DMAX)

            step_ind += 1

        # # need this for 2D where things don't really stay canonical
        # if te_type in ['SU','simple','FU']:
        #     norm = PEPX_GL.norm(tldm,XMAX=XMAX2)
        #     tldm = PEPX_GL.mul(1./norm,tldm)
        #     # print 'tstep norm', norm

        # measure observable
        for obs_key in obs_pepos.keys():
            obs,norm = TLDM_GL.meas_obs(tldm,obs_pepos[obs_key],XMAX=XMAX2,return_norm=True,contract_SL=contract_SL)
            tldm = TLDM_GL.mul(1./norm,tldm)    ## obs is already normalized

            # obs = PEPX_GL.meas_obs(tldm,obs_pepos[obs_key],XMAX=XMAX2,return_norm=False,
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

        # if truncate_run and H_diff > 1.0e-6:
        if truncate_run and (H_diff > 1.0e-6 or np.abs(H_diff) < run_tol):
            print 'energy went up or within run tolerance; truncating TE run', H_diff
            break

        ## ensure canonicalization if possible
        # print 'check time step full', tstep
        if L1 > 1 and L2 == 1:
            gs,ls,axT = PEPX_GL.get_sites( tldm, (0,0), 'o'*(L1-1))
            # PEPX_GL.check_GL_canonical(gs,ls)

            gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=0)
            # print 'check canon'
            PEPX_GL.check_GL_canonical(gs_,ls_)
            tldm = PEPX_GL.set_sites( tldm, (0,0), 'o'*(L1-1), gs_, ls_, axT)

        if L1 == 1 and L2 > 1:
            gs,ls,axT = PEPX_GL.get_sites( tldm, (0,0), 'r'*(L2-1))
            # PEPX_GL.check_GL_canonical(gs,ls)

            gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=1)
            # print 'norm',PEPX_GL.norm(tldm)
            PEPX_GL.check_GL_canonical(gs_,ls_)
            tldm = PEPX_GL.set_sites( tldm, (0,0), 'r'*(L2-1), gs_, ls_, axT)
        # print 'end check time step'

        tstep += 1

        # ## save peps
        # if False:  
        #     np.save('peps_data/runtime_peps_L%dx%d_D%d_gammas.npy' %(L1,L2,DMAX),tldm.view(np.ndarray))
        #     np.save('peps_data/runtime_peps_L%dx%d_D%d_lambdas.npy'%(L1,L2,DMAX),tldm.lambdas)
        
    return obs_t, tldm
### 



def get_trotter_layers(trotterH,version=0):

    return TE_GL.get_trotter_layers(trotterH,version)
