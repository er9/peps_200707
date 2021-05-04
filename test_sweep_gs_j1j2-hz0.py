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
# import PEPX_trotterTE as TE
import PEPX_GL_trotterTE as TE_GL



Ls = (5,5)
L1,L2 = Ls
print 'Ls', Ls

# fdir = 'jjxy_data/'
fdir = 'j1j2_peps_qrDL_data/'
subdir = 'gs_data/'

if not os.path.isdir(fdir):
    os.mkdir(fdir)
if not os.path.isdir(fdir+'peps_states/'):
    os.mkdir(fdir+'peps_states/')
if not os.path.isdir(fdir+subdir):
    os.mkdir(fdir+subdir)

# run info
do_SL = False #True
do_QR = True
do_FU = True
do_2site = False
save_peps = True
save_data = True
scaleX = 1

XMAX2 = 100 # XMAX

# sweep through parameters
for DMAX in [4,5,6]: #[2,3,4,5,6]:

  if DMAX == 2:
    # SU TE
    dt1 = 0.1*-1.j
    totT1 = 10 # 15
    
    # FU TE
    dt2 = 0.1*-1.j
    totT2 = 10

  else:
    # SU TE
    dt1 = 0.1*-1.j
    totT1 = 10 # 15
    
    # FU TE
    dt2 = 0.01*-1.j
    totT2 = 1


  for hz in [0.]: #[0., 0.1, 0.25, 0.5, 0.75, 1.0]:     # [0.,0.05,0.25]:

    hx = 0.

    # for J2 in np.linspace(0.3,0.7,9):
    for J2 in np.linspace(0.3,1.0,15):

      J2 = round(J2,2)

      XMAX = (DMAX**2)*2

      dt1_str = '%04d'%np.rint(np.abs(dt1)*1000)
      dt2_str = '%04d'%np.rint(np.abs(dt2)*1000)
      J2_str = '%04d'%np.rint(J2*100)
      hx_str = '%04d'%np.rint(hx*100)
      hz_str = '%04d'%np.rint(hz*100)

      # file_str = 'L%dx%d_D%d_dt%s_hx%s_zy%s_J2%s'%(L1,L2,DMAX,dt_str,hx_str,hz_str,J2_str)
      fu_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s'%(L1,L2,DMAX,XMAX2,dt2_str,hx_str,hz_str,J2_str)
      su_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s'%(L1,L2,DMAX,XMAX2,dt1_str,hx_str,hz_str,J2_str)

      print 'doing calc for %s: dt %s, J2 %s, hx %s, hz %s'%(fdir,dt2_str,J2_str,hx_str,hz_str)
      print 'doing SL?', do_SL, '; doing QR?', do_QR, 'DMAX %d, XMAX %d, XMAX2 %d'%(DMAX,XMAX,XMAX2)

      ### define trotterH ###
      hs  = [hx,0,hz]
      t1s = [1.,1.,1.]
      t2s = [J2,J2,J2]
      trotterH = Op.t1t2_3body_sum(Ls,hs,t1s,t2s)
      # trotterH  = Op.Heisenberg_sum(Ls,hs,t1s)
      obs_pepos = {'H': trotterH}


      #### do SU, or load SU state if it exists ###
      run_data_SU = {}
      # dts = np.arange(int(np.abs(totT1/dt1))+1)*np.abs(dt1)

      # see if FU already exists
      try: 
          # gammas  = np.load(fdir+'peps_states/'+fu_str+'_FU_X%d_gammas.npy' %XMAX, allow_pickle=True)
          # lambdas = np.load(fdir+'peps_states/'+fu_str+'_FU_X%d_lambdas.npy'%XMAX, allow_pickle=True)
          FU_data = np.loadtxt(fdir+subdir+fu_str+'_FU_X%d_data.txt'%(XMAX))
          if FU_data[-1,0] < totT2+1.0e-12:      continue
          else:                                  print('did not find existing results for long enough time')
      except(IOError):
          print('did not find existing results for run parameters')


      try:
          ## load SU ground state
          gammas  = np.load(fdir+'peps_states/'+su_str+'_SU_gammas.npy' , allow_pickle=True)
          lambdas = np.load(fdir+'peps_states/'+su_str+'_SU_lambdas.npy', allow_pickle=True)
          SU_PEPS_GL = PEPX_GL.PEPX_GL(gammas,lambdas)
          print 'using saved SU state as initial state'
      except(IOError):

          ## perform SU TE
          print 'doing SU'
          try:
              # raise(IndexError)

              ## load FU ground state from lower D
              temp_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s'%(L1,L2,DMAX-1,XMAX2,dt2_str,hx_str,hz_str,J2_str)
              fn_gs = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'peps_states/')], 
                                      temp_str+'_FU_X*_gammas.npy')
              fn_ls = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'peps_states/')],
                                      temp_str+'_FU_X*_lambdas.npy')

              gammas  = np.load(fdir+'peps_states/'+fn_gs[-1], allow_pickle=True)
              lambdas = np.load(fdir+'peps_states/'+fn_ls[-1], allow_pickle=True)
              initPEPS_GL = PEPX_GL.PEPX_GL(gammas,lambdas)
              print 'using saved FU %d -1 state as initial state'%(DMAX)

          except(IndexError):    # list length is 0
              try:
                  ## load SU ground state from lower D
                  temp_str = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s'%(L1,L2,DMAX-1,XMAX2,dt1_str,
                                                                    hx_str,hz_str,J2_str)
                  gammas  = np.load(fdir+'peps_states/'+temp_str+'_SU_gammas.npy' , allow_pickle=True)
                  lambdas = np.load(fdir+'peps_states/'+temp_str+'_SU_lambdas.npy', allow_pickle=True)
                  initPEPS_GL = PEPX_GL.PEPX_GL(gammas,lambdas)
                  print 'using saved SU %d -1 state as initial state'%(DMAX)

              except(IOError):
                  # random initial product state
                  initPEPS_GL = PEPX_GL.random_product_state([[(2,)]*L2]*L1)
                  print 'using random initial product state'

                  # if J2 <= 0.7:     kvec = (1,1)
                  # else:
                  #     if L1 >= L2:  kvec = (1,0)
                  #     else:         kvec = (0,1)

                  # # initial (A)FM product state
                  # initPEPS_GL = PEPX_GL.product_kvec((L1,L2),kvec)
                  # print 'using structured product state', kvec

          obs0 = {}
          for opk in obs_pepos.keys():
              obs = PEPX_GL.meas_obs(initPEPS_GL, obs_pepos[opk], XMAX=XMAX2)
              obs0[opk] = [obs]
          print 'nrg', obs0['H']
          print 'norm', PEPX_GL.norm(initPEPS_GL,XMAX=XMAX2)

          if DMAX==2:  trunc_run = False
          else:        trunc_run = True

          time1a = time.time()
          obs_t1a, pepx1 = TE_GL.run_TE(dt1, totT1, trotterH, initPEPS_GL, DMAX=DMAX, XMAX=XMAX,
                                        XMAX2=XMAX2, obs_pepos=obs_pepos,te_type='SU',qr_reduce=do_QR,
                                        contract_SL=do_SL,truncate_run=trunc_run)

          print '----- D%d, hx%4.3f, J2%4.3f SU -----'%(DMAX,hx,J2)
          print 'run time', np.abs(dt1), totT1, time.time()-time1a
          print 'nrg', obs_t1a['H'][-1]/L1/L2
          print [np.linalg.norm(m) for idx,m in np.ndenumerate(pepx1)]
          print [pepx1.lambdas[idx] for idx in np.ndindex((L1,L2))]
          print 'norm', PEPX_GL.norm(pepx1), '\n'
          SU_PEPS_GL = pepx1

          for opk in obs_pepos.keys():
              run_data_SU[opk] = obs0[opk] + obs_t1a[opk]

          if save_peps:
              np.save(fdir+'peps_states/'+su_str+'_SU_gammas.npy' ,pepx1.view(np.ndarray))
              np.save(fdir+'peps_states/'+su_str+'_SU_lambdas.npy',pepx1.lambdas)

          if save_data:
              dts = np.arange(len(run_data_SU['H']))*np.abs(dt1)
              SU_data = [dts,run_data_SU['H']]
              header_str = 'time H'
              for opk in obs_pepos.keys():
                  if opk != 'H':
                      SU_data += [run_data_SU[opk]]
                      header_str += ' ' + opk

              np.savetxt(fdir+subdir+su_str+'_SU_data.txt',np.real(np.array(SU_data)).T,header=header_str)


      ### do full update starting from SU_PEPS_GL ###
      if do_FU: 

          run_success = False

          while not run_success:

              print 'doing FU, D', DMAX, 'X', XMAX
              run_data_FU = {}
              dts = np.arange(int(np.abs(totT2/dt2))+1)*np.abs(dt2)

              obs0 = {}
              for opk in obs_pepos.keys():
                  obs = PEPX_GL.meas_obs(SU_PEPS_GL, obs_pepos[opk], XMAX=XMAX2)
                  obs0[opk] = [obs]
              print 'nrg', obs0['H']
              print 'norm', PEPX_GL.norm(SU_PEPS_GL,XMAX=XMAX2)

              try:
                  time2a = time.time()
                  obs_t2a, pepx2 = TE_GL.run_TE(dt2,totT2,trotterH,SU_PEPS_GL,DMAX=DMAX,XMAX=XMAX,
                                                XMAX2=XMAX2,obs_pepos=obs_pepos,te_type='FU',
                                                qr_reduce=do_QR, contract_SL=do_SL,alsq_2site=do_2site,scaleX=scaleX)
      
      
                  print '----- D%d, hx%4.3f, J2%4.3f FU -----'%(DMAX,hx,J2)
                  print 'run time', np.abs(dt2), totT2, time.time()-time2a
                  print 'nrg', obs_t2a['H'][-1]/L1/L2
                  print [np.linalg.norm(m) for idx,m in np.ndenumerate(pepx2)]
                  print [pepx2.lambdas[idx] for idx in np.ndindex((L1,L2))]
                  print 'norm', PEPX_GL.norm(pepx2),'\n'

                  run_success = True

                  for opk in obs_pepos.keys():
                      run_data_FU[opk] = obs0[opk] + obs_t2a[opk]

                  if save_peps:
                      np.save(fdir+'peps_states/'+fu_str+'_FU_X%d_gammas.npy' %(XMAX),pepx2.view(np.ndarray))
                      np.save(fdir+'peps_states/'+fu_str+'_FU_X%d_lambdas.npy'%(XMAX),pepx2.lambdas)

                  if save_data:
                      dts = np.arange(len(run_data_FU['H']))*np.abs(dt2)
                      FU_data = [dts,run_data_FU['H']]
                      header_str = 'time H'
                      for opk in obs_pepos.keys():
                          if opk != 'H':
                              FU_data += [run_data_FU[opk]]
                              header_str += ' ' + opk

                      np.savetxt(fdir+subdir+fu_str+'_FU_X%d_data.txt'%(XMAX),
                                 np.real(np.array(FU_data)).T,header=header_str)

              except(RuntimeWarning,RuntimeError):
                  print 'FU failed, D %d, X %d, J2 %s'%(DMAX,XMAX,str(J2))
                  print 'raising XMAX by 2*DMAX**2'
                  XMAX = XMAX + (DMAX**2)*2
                  if XMAX > 250:
                      print 'exiting. max X'
                      exit()
                  # raise(RuntimeError)
          

