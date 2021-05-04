import numpy as np
import time
import matplotlib.pyplot as plt

import tens_fcts as tf
import MPX  
import PEPX
import PEPX_GL
import PEPS
import PEPS_GL
import PEPS_env as ENV
import PEP0_env as ENV0
import PEPS_GL_env_nolam as ENV_GL
import Operator_2D as Op
import PEPX_GL_trotterTE as TE_GL



paulis = Op.paulis

# def collapse_to_prod(pepx,trotterH,axis=[0,0,1],XMAX=100):
#     ''' collaspe wavefunction to product state '''
# 
#     L1, L2 = pepx.shape
#     # print pepx.phys_bonds
#     assert(np.all([ db == (2,) for idx,db in np.ndenumerate(pepx.phys_bonds)])),'only implemented for phys bonds db=2'
# 
#     new_pepx = np.empty((L1,L2),dtype=np.object)
#     occ = np.empty((L1,L2),dtype=np.object)
# 
#     # only works for dbs=2
#     proj_ax = axis[0]*paulis['SX'] + axis[1]*paulis['SY'] + axis[2]*paulis['SZ']
#     if np.abs(axis[1]) > 1.0e-8:
#         proj1 = PEPX.empty( [[(2,2)]] ,1, dtype=np.complex128)
#         proj2 = PEPX.empty( [[(2,2)]] ,1, dtype=np.complex128)
#     else: 
#         proj1 = PEPX.empty( [[(2,2)]] ,1)
#         proj2 = PEPX.empty( [[(2,2)]] ,1)
#         proj_ax = np.real(proj_ax)
# 
#     proj1[0,0][0,0,0,0,:,:] = 1./2*np.eye(2) + proj_ax
#     proj2[0,0][0,0,0,0,:,:] = 1./2*np.eye(2) - proj_ax
# 
# 
#     # calculate environment
#     envIs = ENV.get_boundaries( np.conj(pepx), pepx, 'I', L1, XMAX=XMAX)  # list of len 1
#     envOs = ENV.get_boundaries( np.conj(pepx), pepx, 'O', 0,  XMAX=XMAX)  # list of len L+1
# 
#     for i in range(L1):       # rows
#         senvRs = ENV.get_subboundaries(envIs[i],envOs[i+1],np.conj(pepx[i:i+1,:]),pepx[i:i+1,:],'R',0, XMAX=XMAX)
#         senvLs = ENV.get_subboundaries(envIs[i],envOs[i+1],np.conj(pepx[i:i+1,:]),pepx[i:i+1,:],'L',L2,XMAX=XMAX)
#         
#         for j in range(L2): 
#             bi = envIs[i][j:j+1]
#             bo = envOs[i+1][j:j+1]
#             bl = senvLs[j]
#             br = senvRs[j+1] 
# 
#             pepx_sub = pepx[i:i+1,j:j+1]
#             pepx1 = PEPX.dot(proj1,pepx_sub)
#             pepx2 = PEPX.dot(proj2,pepx_sub)
# 
#             p1 = ENV.embed_sites_ovlp(np.conj(pepx1),pepx1,[bl,bi,bo,br],XMAX=XMAX)
#             p2 = ENV.embed_sites_ovlp(np.conj(pepx2),pepx2,[bl,bi,bo,br],XMAX=XMAX)
#  
#             if not(np.abs(p1+p2-1) < 1.0e-8):
#                 print 'error in metts projectors, %f %f'%(p1,p2)
#                 print 'norm', ENV.embed_sites_ovlp(np.conj(pepx_sub),peps_sub,[bl,bi,bo,br])
#                 exit()
# 
#             choose_1 = np.random.random() < p1
#             if choose_1:
#                 new_tens = PEPX.mul(1./np.sqrt(p1),pepx1)
#                 occ[i,j] = (0,)
#             else:
#                 # pepx2 = PEPX.dot(proj2,pepx_sub)
#                 # new_tens = PEPX.mul(1./np.sqrt(1-p1),pepx2)
#                 new_tens = PEPX.mul(1./np.sqrt(p2),pepx2)
#                 occ[i,j] = (1,)
#  
# 
#             new_pepx[i,j] = new_tens[0,0]
#             # test[i,j] = new_tens
#             new_norm = ENV.embed_sites_norm(new_tens,[bl,bi,bo,br],XMAX=XMAX)
#             if np.abs(new_norm -1) > 1.0e-8:
#                 print 'update tens error', new_norm
# 
#     new_ = PEPX.PEPX(new_pepx,phys_bonds=pepx.phys_bonds)
#     new_norm = PEPX.norm(new_)
#     if np.abs(new_norm - 1) > 1.0e-8:
#         print 'cps not normalized', new_norm
#         # exit()
# 
#     new_ = PEPX.product_peps(pepx.phys_bonds,occ)
# 
#     return new_


# @profile
def collapse_to_prod(pepx_,axis=[0,0,1],XMAX=100, envs=None):
    ''' collaspe wavefunction to product state '''

    pepx = pepx_.copy()

    L1, L2 = pepx.shape
    # print pepx.phys_bonds
    assert(np.all([ db == (2,) for idx,db in np.ndenumerate(pepx.phys_bonds)])),'only implemented for phys bonds db=2'

    # only works for dbs=2
    proj_ax = axis[0]*paulis['SX'] + axis[1]*paulis['SY'] + axis[2]*paulis['SZ']
    if np.abs(axis[1]) > 1.0e-8:
        proj1 = PEPX_GL.empty( [[(2,2)]] ,1, dtype=np.complex128)
        proj2 = PEPX_GL.empty( [[(2,2)]] ,1, dtype=np.complex128)
    else: 
        proj1 = PEPX_GL.empty( [[(2,2)]] ,1)
        proj2 = PEPX_GL.empty( [[(2,2)]] ,1)
        proj_ax = np.real(proj_ax)

    proj1[0,0][0,0,0,0,:,:] = 1./2*np.eye(2) + proj_ax
    proj2[0,0][0,0,0,0,:,:] = 1./2*np.eye(2) - proj_ax


    # calculate environment
    envIs = ENV_GL.get_boundaries( np.conj(pepx), pepx, 'I', 0, XMAX=XMAX)  # list of len 1
    envOs = ENV_GL.get_boundaries( np.conj(pepx), pepx, 'O', 0, XMAX=XMAX)  # list of len L+1

    for i in range(L1):       # rows
        senvLs = ENV_GL.get_subboundaries(envIs[i],envOs[i+1],np.conj(pepx[i:i+1,:]),pepx[i:i+1,:],'L',0,XMAX=XMAX)
        senvRs = ENV_GL.get_subboundaries(envIs[i],envOs[i+1],np.conj(pepx[i:i+1,:]),pepx[i:i+1,:],'R',0,XMAX=XMAX)
        
        for j in range(L2): 
            bi = envIs[i][j:j+1]
            bo = envOs[i+1][j:j+1]
            bl = senvLs[j]
            br = senvRs[j+1] 

            pepx_sub = pepx[i:i+1,j:j+1]
            full_norm = ENV_GL.embed_sites_norm(pepx_sub,[bl,bi,bo,br])
            pepx_sub = PEPX_GL.mul(1./full_norm,pepx_sub)
            # print 'full norm', full_norm

            pepx1 = PEPX_GL.dot(proj1,pepx_sub)
            p1 = ENV_GL.embed_sites_ovlp(np.conj(pepx1),pepx1,[bl,bi,bo,br])
            # pepx2 = PEPX_GL.dot(proj2,pepx_sub)
            # p2 = ENV_GL.embed_sites_ovlp(np.conj(pepx2),pepx2,[bl,bi,bo,br])
 
            # if not(np.abs(p1+p2-1) < 1.0e-8):
            #     print 'error in metts projectors, %f %f'%(p1,p2)
            #     print 'norm', ENV.embed_sites_ovlp(np.conj(pepx_sub),peps_sub,[bl,bi,bo,br])
            #     exit()


            choose_1 = np.random.random() < p1
            if choose_1:
                new_pepx = PEPX_GL.mul(1./np.sqrt(p1),pepx1)
            else:
                pepx2  = PEPX_GL.dot(proj2,pepx_sub)
                new_pepx = PEPX_GL.mul(1./np.sqrt(1-p1),pepx2)


            new_norm = ENV_GL.embed_sites_norm(new_pepx,[bl,bi,bo,br])
            if np.abs(new_norm -1) > 1.0e-8:
                print 'METTS new cps norm not 1:', new_norm
            pepx[i,j] = new_pepx[0,0]*1./new_norm
            # for x in range(4):   pepx = set_bond(pepx,(i,j),x,new_pepx.lambdas[0,0,x]   # shouldn't change tho

            # print len(senvLs), senvLs[-1].shape, [m.shape for m in senvLs[-1]], [m.shape for m in bl]
            e_mpo,err = ENV_GL.get_next_subboundary_L(senvLs[-1],bi[0],np.conj(pepx[i:i+1,j]),pepx[i:i+1,j],bo[0],XMAX)
            senvLs.append(e_mpo)

        e_mpo, err = ENV_GL.get_next_boundary_I(np.conj(pepx[i,:]),pepx[i,:],envIs[-1],XMAX)
        envIs.append(e_mpo)

    new_norm = np.sqrt( ENV_GL.ovlp_from_bound(senvLs[-1]) )
    if np.abs(new_norm - 1) > 1.0e-8:
        print 'cps not normalized', new_norm
        pepx = PEPX_GL.mul(1./new_norm,pepx)


    
    ## ensure canonicalization if possible
    if L1 > 1 and L2 == 1:
        gs,ls,axT = PEPX_GL.get_sites( pepx, (0,0), 'o'*(L1-1))
        # PEPX_GL.check_GL_canonical(gs,ls)

        gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=0)
        PEPX_GL.check_GL_canonical(gs_,ls_)
        pepx = PEPX_GL.set_sites( pepx, (0,0), 'o'*(L1-1), gs_, ls_, axT)

    if L1 == 1 and L2 > 1:
        gs,ls,axT = PEPX_GL.get_sites( pepx, (0,0), 'r'*(L2-1))
        # PEPX_GL.check_GL_canonical(gs,ls)

        gs_,ls_ = PEPX_GL.canonicalize_GL_list(gs,ls,direction=1)
        PEPX_GL.check_GL_canonical(gs_,ls_)
        pepx = PEPX_GL.set_sites( pepx, (0,0), 'r'*(L2-1), gs_, ls_, axT)
    # print 'end check time step'


    # print 'check canon metts prod'
    # if L1 > 1 and L2 == 1:
    #     gs,ls,axT = PEPX_GL.get_sites( pepx, (0,0), 'o'*(L1-1))
    #     PEPX_GL.check_GL_canonical(gs,ls)

    return pepx


# @profile
def run_metts(Ls,trotterH,beta,dt,num_samples,obs_pepos={},initPEPS=None,DMAX=5,XMAX=100,te_type='SU'):

    L1,L2 = Ls
    obs_steps = {}

    # rand occ
    rand_str = np.random.rand(L1,L2)
    rand_occ = np.rint(rand_str).astype(int)

    initPEPS = PEPX_GL.product_peps([[(2,)]*L2]*L1,rand_occ)   # dp=2, random initial occ
    for obs in obs_pepos.keys():
        obs_steps[obs] = [PEPX_GL.meas_obs(initPEPS, obs_pepos[obs])]
    print 'metts init norm', PEPX_GL.norm(initPEPS), obs_steps['HNN']
        
    # if np.isreal(dt):   dt = -1.j*dt

    n_th = 0
    while n_th < num_samples:

        obs_t, beta_peps = TE_GL.run_TE(dt, beta, trotterH, initPEPS, DMAX=DMAX, XMAX=XMAX, obs_pepos={},
                                        te_type=te_type)
        # should change this fct to not calc any obs_t

        # print [m.shape for idx, m in np.ndenumerate(beta_peps)]

        for obs in obs_pepos.keys():
            # temp = PEPX.meas_obs(beta_peps, obs_pepos[obs])
            obs_steps[obs] += [PEPX_GL.meas_obs(beta_peps, obs_pepos[obs])]
        print beta, n_th, obs_steps['HNN'][-1], PEPX_GL.norm(beta_peps)
        if obs_steps['HNN'][-1] < -2.75:
            print 'energy too low'
            exit()

        # alternate between axis = 100 and 001 for each thermal step as in Miles's paper
        ax = [(n_th)%2,0,(n_th+1)%2]
        # ax = [0,0,1]
        initPEPS = collapse_to_prod(beta_peps,axis=ax,XMAX=100)  #XMAX)  

        n_th += 1

    return obs_steps, initPEPS


# @profile
def TE_METTS(Ls,dt,totT,trotterH,obs_pepos={},DMAX=5,XMAX=100,every_x=10,num_samples=50,max_sweeps=10,
             te_type='SU',return_data=False):
    ''' do METTS for each time step until totT '''    

    betas = []
    avg_obs_t = {}
    all_obs_t = {}

    tstep = 0
    while tstep < totT/np.abs(dt)/every_x:
        not_converged = True
        # print 'beta',tstep, totT/np.abs(dt)/every_x
        beta = np.abs(dt)*every_x*(tstep+1)
        betas.append(beta)

        therm_step = 0
        while not_converged and therm_step < max_sweeps:
            # print 'thermal sampling', therm_step
            try:
                obs_steps, cps = run_metts(Ls,trotterH,beta,dt,num_samples,obs_pepos,initPEPS=cps,DMAX=DMAX,
                                           XMAX=XMAX,te_type=te_type)
            except(NameError):  # cps not defined
                obs_steps, cps = run_metts(Ls,trotterH,beta,dt,num_samples,obs_pepos,DMAX=DMAX,XMAX=XMAX,
                                           te_type=te_type)

            obs_notconverged = []
            for obs_key in obs_pepos.keys():
                avg_step = np.mean(obs_steps[obs_key])
                if tstep == 0 and therm_step == 0:
                    old_avg = np.inf
                    avg_obs_t[obs_key] = [avg_step]
                    if return_data:   all_obs_t[obs_key] = [obs_steps[obs_key]]
                elif therm_step == 0:
                    old_avg = np.inf
                    avg_obs_t[obs_key] += [avg_step]
                    if return_data:   all_obs_t[obs_key] += [obs_steps[obs_key]]
                else:
                    old_avg  = avg_obs_t[obs_key][-1]   # for current t_step
                    avg_obs_t[obs_key][-1] = (therm_step*old_avg+avg_step)/(therm_step+1)
                    if return_data:   all_obs_t[obs_key][-1] += obs_steps[obs_key]

                obs_notconverged += [np.abs((old_avg-avg_obs_t[obs_key][-1])/avg_obs_t[obs_key][-1]) > 1.0e-3]

            # print obs_notconverged

            therm_step += 1
            not_converged = np.any(obs_notconverged)

        print 'metts: num samples', therm_step*num_samples, 'beta', beta # tstep*dt*every_x
        tstep += 1



    if return_data:    return betas, avg_obs_t, all_obs_t
    else:              return betas, avg_obs_t

       


        





