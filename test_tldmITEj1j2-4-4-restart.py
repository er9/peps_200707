import os
import fnmatch 

import numpy as np
import time
import matplotlib
# matplotlib.use('Agg')
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
import PEPX_GL_trotterTE as TE_GL

import TLDM_GL
import TLDM_GL_env_nolam as ENV_DM
import TLDM_trotterTE as TE_DM

import TimeEvolution as TE1
# import PEPX_trotterTE as TE


##############################################
#### thermal state properties SU vs FU #######
##############################################

dt = 0.1*-1.j
totT = 7.5 #15      # --> temp T = 1/totT

restart = True
restart_beta = 14.0 # 10.0  # beta/2 * 2
# [0.1,0.2,0.3,0.5,1.0,1.5,2.0,3.0,5.0,7.5]
target_betas = [7.5] #10.0,15.0]:  # beta/2

Ls = (5,5)  #(3,3)
L1,L2 = Ls
print 'lattice size', Ls

fdir = 'j1j2_tldm_qrDL_data/' #DL_data/'
subdir = ''

if not os.path.isdir(fdir):
    os.mkdir(fdir)
if not os.path.isdir(fdir+'tldm_states/'):
    os.mkdir(fdir+'tldm_states/')
if not os.path.isdir(fdir+subdir):
    os.mkdir(fdir+subdir)

# obs_pepos = {'mZ':[[('SZ',)]*(L1*L2),site_inds],'mX':[[('SX',)]*(L1*L2),site_inds],
#              'mXX':[[('SX','SX')]*((L1-1)*(L2-1)*2),diag_inds],'mYY':[[('SY','SY')]*((L1-1)*(L2-1)*2),diag_inds]}
obs_pepos = {}

# run info
do_SU = False #True
do_FU = True

do_SL = False #True
qr_reduce = True #False
do_alsq_2site = False #True
save_data = True
save_tldm = True


hx,hz = 0,0
XMAX2 = 100

Ds = [5] # [4,5,6] #[2,3,4,5,6]
for DMAX in Ds:

  # for XMAX in [25]:  # [25]:

  for J2 in [0.9]:  #[0.3,0.35,0.4]: #,0.05,0.25]:

    XMAX = (DMAX**2)*3

    dt_str = '%04d'%int(np.abs(dt)*1000)
    J2_str = '%04d'%int(J2*100)
    hx_str = '%04d'%int(hx*100)
    hz_str = '%04d'%int(hz*100)

    file_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s'%(L1,L2,DMAX,XMAX2,dt_str,hx_str,hz_str,J2_str)

    print 'doing calc for dt %s, J2 %s, hx %s, hz %s'%(dt_str,J2_str,hx_str,hz_str)
    print 'doing SL?', do_SL, '; doing QR?', qr_reduce, 'DMAX %d, XMAX %d, XMAX2 %d'%(DMAX,XMAX,XMAX2)


    ### define trotterH ###
    hs  = [hx,0,hz]
    t1s = [1.,1.,1.]
    t2s = [J2,J2,J2]
    # hs  = [0.,0.,0.]
    # t1s = [1.,1.,1.]
    # t2s = [0.,0.,0.]
    trotterH = Op.t1t2_3body_sum(Ls,hs,t1s,t2s)
    obs_pepos['H'] = trotterH

    ### define normalized identity ###
    initDM = TLDM_GL.eye([[(2,)]*L2]*L1)
    norm_val = TLDM_GL.norm(initDM)
    print norm_val
    initDM = TLDM_GL.mul( 1./norm_val, initDM )
    print TLDM_GL.norm(initDM)


    ### data collection ###
    if restart:
        betas = [restart_beta]
        target_betas = [tb for tb in target_betas if tb > restart_beta/2] 
        beta_str_0 = '%04d'%(int(betas[-1]*100))

        if do_SU:
            tldm_gammas  = np.load(fdir+'tldm_states/'+file_str+'_beta%s_SU_gammas.npy' %(beta_str_0))
            tldm_lambdas = np.load(fdir+'tldm_states/'+file_str+'_beta%s_SU_lambdas.npy'%(beta_str_0))
            initDM = TLDM_GL.TLDM_GL(tldm_gammas,tldm_lambdas)
        if do_FU:
            fn_gs = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'tldm_states/')], file_str+'_beta%s_FU_'%(beta_str_0)+'*_gammas.npy')
            fn_ls = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'tldm_states/')], file_str+'_beta%s_FU_'%(beta_str_0)+'*_lambdas.npy')

            for fnum in range(len(fn_gs)):
                fstr = fn_gs[fnum][:-11]
                print 'fstr', fstr
                X_ind = fstr.find('X',-5)
                Xstr = fstr[X_ind+1:]
                print 'Xstr', Xstr
                XMAX = max(int(Xstr),XMAX)

            tldm_gammas  = np.load(fdir+'tldm_states/'+file_str+'_beta%s_FU_X%d_gammas.npy' %(beta_str_0,XMAX))
            tldm_lambdas = np.load(fdir+'tldm_states/'+file_str+'_beta%s_FU_X%d_lambdas.npy'%(beta_str_0,XMAX))
            initDM = TLDM_GL.TLDM_GL(tldm_gammas,tldm_lambdas)

        print('loaded beta tldm', betas[-1], XMAX)

    else:
        print('starting from beta = 0')
        betas = [0.0]

    op_keys = obs_pepos.keys()
    print 'op keys', op_keys
    run_data_FU = {}
    run_data_SU = {}
    for opk in op_keys:
        obs0 = TLDM_GL.meas_obs(initDM, obs_pepos[opk], XMAX=100)
        run_data_SU[opk] = [obs0]
        run_data_FU[opk] = [obs0]

    # fig1,ax1 = plt.subplots()
    # fig2,ax2 = plt.subplots()
    # legend = []

    tldm1 = initDM
    tldm2 = initDM

    for target_beta in target_betas:
    # for target_beta in [0.1,0.3,0.5,1.0,1.5,2.0,3.0,5.0,7.5]: #10.0,15.0]:
    # for target_beta in [0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0,10.0,15.0]:
    # for d_beta in np.abs(np.diff([0.0, 0.1,0.25,0.5,0.75,1.0,1.5,2.0,3.0,5.0,10.0,15.0])):

        time1 = time.time()

        d_beta = target_beta - betas[-1]/2
        nsteps = int(round(abs(d_beta/dt),2))
        db1 = np.arange(nsteps)*np.abs(dt)*2 + betas[-1] + np.abs(dt)*2
        betas = np.append( betas, db1 )
        beta_str = '%04d'%(int(db1[-1]*100))
  
        # ########## run 1 (SU) #################
        if do_SU:
            print 'doing SU, D', DMAX

            time1a = time.time()
            obs_t1a, tldm1 = TE_DM.run_TE(dt,d_beta,trotterH,tldm1,DMAX=DMAX,XMAX2=XMAX2,obs_pepos=obs_pepos,
                                         build_env=False,te_type='SU',qr_reduce=qr_reduce,contract_SL=do_SL)
            print 'run time', time.time()-time1a

            # observables have overlap of last el of A and first el of previous run
            print 'nrg', db1[-1], obs_t1a['H'][-1]/L1/L2
            print 'norm', TLDM_GL.norm(tldm1)

            for opk in op_keys:
                run_data_SU[opk] = run_data_SU[opk] + obs_t1a[opk]

            if save_tldm:
                np.save(fdir+'tldm_states/'+file_str+'_beta%s_SU_gammas.npy'%(beta_str), 
                        tldm1.view(np.ndarray))
                np.save(fdir+'tldm_states/'+file_str+'_beta%s_SU_lambdas.npy'%(beta_str),
                        tldm1.lambdas)

            if save_data:
                header_str = 'beta '
                for opk in op_keys:   header_str += (opk+' ')
  
                SU_data = [betas,run_data_SU['H']]
                header_str = 'time H'
                for opk in obs_pepos.keys():
                    if opk != 'H':
                        SU_data += [run_data_SU[opk]]
                        header_str += ' ' + opk
                np.savetxt(fdir+subdir+file_str+'_SU_data.txt',np.real(np.array(SU_data)).T,header=header_str)
  
        # ########## run 1 (FU) #################
        ## heat capacity ##
        if do_FU:

            run_success = False

            while not run_success:

                print 'doing FU, D', DMAX, 'X', XMAX

                time2a = time.time()
                try:
                    obs_t2a, tldm2 = TE_DM.run_TE(dt,d_beta,trotterH,tldm2,DMAX=DMAX,XMAX=XMAX,XMAX2=XMAX2,
                                                  obs_pepos=obs_pepos,te_type='FU',
                                                  qr_reduce=qr_reduce,contract_SL=do_SL,alsq_2site=do_alsq_2site)
                    print 'run time', time.time()-time2a

                    print 'nrg', betas[-1], obs_t2a['H'][-1]/L1/L2
                    print 'norm', TLDM_GL.norm(tldm2)

                    for opk in op_keys:
                        run_data_FU[opk] = run_data_FU[opk] + obs_t2a[opk]

                    if save_tldm:
                        np.save(fdir+'tldm_states/'+file_str+'_beta%s_FU_X%d_gammas.npy'%(beta_str,XMAX),
                                tldm2.view(np.ndarray))
                        np.save(fdir+'tldm_states/'+file_str+'_beta%s_FU_X%d_lambdas.npy'%(beta_str,XMAX),
                                tldm2.lambdas)

                    run_success = True

                except(RuntimeError,RuntimeWarning):
                    print 'error occurred (likely in run_TE) for DMAX %d'%DMAX
                    print 'increasing XMAX by 2'
                    XMAX = XMAX + 50
                    if XMAX > 250:
                        print 'exiting. max X'
                        exit()


            if save_data:

                FU_data = [betas,run_data_FU['H']]
                header_str = 'time H'
                for opk in obs_pepos.keys():
                    if opk != 'H':
                        FU_data += [run_data_FU[opk]]
                        header_str += ' ' + opk

                try:
                    orig_data = np.loadtxt(fdir+subdir+file_str+'_FU_X%d_data.txt'%(XMAX))
                    orig_data = orig_data.T
                    ind_cut = int(betas[0]/2/np.abs(dt))
                    print('orig_data ind cut',ind_cut,orig_data.shape,orig_data[0,ind_cut],betas[0])
                    assert( np.abs(orig_data[0,ind_cut] - betas[0]) < 1.0e-12, 'not the correct cut ind, %d, %d'%(ind_cut, orig_data[0,ind_cut]) )

                    for j in range(len(FU_data)):
                        FU_data[j] = np.append(orig_data[j,:ind_cut], FU_data[j])

                    np.savetxt(fdir+subdir+file_str+'_FU_X%d_data.txt'%(XMAX),np.real(np.array(FU_data)).T,header=header_str)

                except:
                    print('error saving separate file')
                    np.savetxt(fdir+subdir+file_str+'_FU_X%d_data_restart.txt'%(XMAX),np.real(np.array(FU_data)).T,header=header_str)


    try:     print 'SU energy', run_data_SU['H'][-1]/L1/L2
    except:  pass

    try:     print 'FU energy', run_data_FU['H'][-1]/L1/L2
    except:  pass



