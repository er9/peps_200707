import fnmatch
import os
import time

import numpy as np
import matplotlib.pyplot as plt

import tens_fcts as tf
import MPX
import PEPX
import PEPS
import PEPS_env as ENV
import PEP0_env as ENV0
import Operator_1D as Op1
import Operator_2D as Op

import PEPX_GL
import PEPS_GL
import PEPS_GL_env_nolam as ENV_GL

import TimeEvolution as TE1
import PEPX_trotterTE as TE
import PEPX_GL_trotterTE as TE_GL


if True:

    # SU TE
    dt1 = 0.1*-1.j
    totT1 = 10

    # FU TE
    dt2 = 0.05*-1.j
    totT2 = 5

    Ls = (5,5)
    L1,L2 = Ls
    print 'Ls', Ls

    # fdir = 'jjxy_data/'
    fdir = 'jjxy_peps_test/swap/'
    subdir = 'Ds_data/'

    if not os.path.isdir(fdir):
        os.mkdir(fdir)
    if not os.path.isdir(fdir+'peps_states/'):
        os.mkdir(fdir+'peps_states/')
    if not os.path.isdir(fdir+subdir):
        os.mkdir(fdir+subdir)

    # run info
    do_SL = True
    do_QR = False
    save_peps = True #False
    save_data = True #False

    oldX2 = 100

    # sweep through parameters
    for DMAX in [2,3,4,5,6]:

        for hx in [0.]:     # [0.,0.05,0.25]:

            for J2 in [0.7]:  # np.linspace(0,1,11):

                XMAX  = DMAX**2*3  # (DMAX-1)*50  #[100,150,200,250,300]:
                XMAX2 = XMAX

                hz = hx
                dt_str = '%04d'%np.rint(np.abs(dt1)*1000)
                J2_str = '%04d'%np.rint(J2*100)
                hx_str = '%04d'%np.rint(hx*100)
                hz_str = '%04d'%np.rint(hz*100)
    
                # file_str = 'L%dx%d_D%d_dt%s_hx%s_zy%s_J2%s'%(L1,L2,DMAX,dt_str,hx_str,hz_str,J2_str)
                file_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s_DL'%(L1,L2,DMAX,XMAX2,dt_str,hx_str,hz_str,J2_str)
    
                print 'doing calc for %s: dt %s, J2 %s, hx %s, hz %s'%(fdir,dt_str,J2_str,hx_str,hz_str)
                print 'doing SL?', do_SL, '; doing QR?', do_QR, 'DMAX %d, XMAX %d, XMAX2 %d'%(DMAX,XMAX,XMAX2)

                ### define trotterH ###
                hs  = [hx,0,hz]
                t1s = [1.,1.,0.]
                t2s = [J2,J2,0.]
                # hs  = [0,0,0]
                # t1s = [1.,1.,1.]
                # t2s = [J2,J2,J2]

                H_nn   = Op.Heisenberg_sum(Ls,hs,t1s)
                H_sw1  = Op.NNN_swap_row(Ls,0)
                H_sw2  = Op.NNN_swap_row(Ls,1)
                H_nnn1 = Op.NNN_t2_row(Ls,t2s,0)
                H_nnn2 = Op.NNN_t2_row(Ls,t2s,1)
                trotter_list = [H_nn, H_sw1, H_nnn1, H_sw1, H_sw2, H_nnn2, H_sw2]
                # DMAXs = [DMAX, -1, -1, DMAX, -1, -1, DMAX]
                DMAXs = [DMAX]*len(trotter_list)

                trotterH = Op.t1t2_3body_sum(Ls,hs,t1s,t2s)
                # trotterH = Op.Heisenberg_sum(Ls,[0,0,0],[1.,1.,1.])
                obs_pepos = {'H': trotterH}


                #### skip SU -- do FU directly ####
                #### do SU, or load SU state if it exists ###
                dts = np.arange(int(np.abs(totT1/dt1))+1)*np.abs(dt1)

                ## load existing FU TE gs
                print 'loading FU gs'
                try:
                    ## load FU ground state from lower D
                    temp_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s'%(L1,L2,DMAX-1,oldX2,dt_str,hx_str,hz_str,J2_str)

                    fn_gs = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'peps_states/')], temp_str+'_FU_X*_gammas.npy')
                    fn_ls = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'peps_states/')], temp_str+'_FU_X*_lambdas.npy')

                    gammas  = np.load(fdir+'peps_states/'+fn_gs[-1], allow_pickle=True)
                    lambdas = np.load(fdir+'peps_states/'+fn_ls[-1], allow_pickle=True)
                    initPEPS_GL = PEPX_GL.PEPX_GL(gammas,lambdas)
                    print 'using saved FU %d -1 state as initial state'%(DMAX)
                    print 'fstr', fn_gs[-1]

                except(IndexError):    # list length is 0
                    try:
                        ## load SU ground state from lower D
                        temp_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s'%(L1,L2,2,oldX2,dt_str,hx_str,hz_str,J2_str)
                        gammas  = np.load(fdir+'peps_states/'+temp_str+'_SU_gammas.npy' , allow_pickle=True)
                        lambdas = np.load(fdir+'peps_states/'+temp_str+'_SU_lambdas.npy', allow_pickle=True)
                        initPEPS_GL = PEPX_GL.PEPX_GL(gammas,lambdas)
                        print 'using saved SU D2 state as initial state'

                    except(IOError):
                        print 'did not find', fdir+'peps_states/'+temp_str
                        # random initial product state
                        initPEPS_GL = PEPX_GL.random_product_state([[(2,)]*L2]*L1)
                        print 'using random initial product state'


                ### do full update starting from D-1 FU_PEPS_GL ###
                print 'doing FU'
                run_data_FU = {}
                dts = np.arange(int(np.abs(totT2/dt2))+1)*np.abs(dt2)

                obs0 = {}
                for opk in obs_pepos.keys():
                    obs = PEPX_GL.meas_obs(initPEPS_GL, obs_pepos[opk], XMAX=XMAX2)
                    obs0[opk] = [obs]
                print 'init nrg', obs0['H'][0]/L1/L2
                print 'init norm', PEPX_GL.norm(initPEPS_GL,XMAX=XMAX2)

                try:
                    time2a = time.time()
                    obs_t2a, pepx2 = TE_GL.run_TE_listH(dt2,totT2,trotter_list,initPEPS_GL,DMAXs=DMAXs,XMAXs=[XMAX],
                                                        XMAX2s=[XMAX2],obs_pepos=obs_pepos,te_type='FU',
                                                        qr_reduce=do_QR,contract_SL=do_SL)

                
                    print '----- D%d, hx%4.3f, J2%4.3f FU -----'%(DMAX,hx,J2)
                    print 'run time', np.abs(dt2), totT2, time.time()-time2a
                    print 'nrg', obs_t2a['H'][-1]/L1/L2
                    print 'norm', PEPX_GL.norm(pepx2),'\n'

                    for opk in obs_pepos.keys():
                        run_data_FU[opk] = obs0[opk] + obs_t2a[opk]

                    if save_peps:
                        np.save(fdir+'peps_states/'+file_str+'_FU_X%d_gammas.npy' %(XMAX),pepx2.view(np.ndarray))
                        np.save(fdir+'peps_states/'+file_str+'_FU_X%d_lambdas.npy'%(XMAX),pepx2.lambdas)

                    if save_data:
                        FU_data = [dts,run_data_FU['H']]
                        header_str = 'time H'
                        for opk in obs_pepos.keys():
                            if opk != 'H':
                                FU_data += [run_data_FU[opk]]
                                header_str += ' ' + opk

                        np.savetxt(fdir+subdir+file_str+'_FU_X%d_data.txt'%(XMAX),np.real(np.array(FU_data)).T,header=header_str)

                    run_done = True
                    oldX2 = XMAX2

                except(RuntimeError):

                    run_done = False
                    XMAX += 50

