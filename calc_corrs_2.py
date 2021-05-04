import pickle
import os
import fnmatch
import numpy as np

import PEPX
import PEPX_GL
import TLDM_GL
import Operator_2D as Op

import meas_state as meas


def save_obj(obj,fdir,fstr):
    with open(fdir+fstr+'.pkl','wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(fdir,fstr):
    with open(fdir+fstr+'.pkl','rb') as f:
        return pickle.load(f)



def get_corrs(fdir,fstr,corr_type,do_mp=True,is_tldm=False,subdir=None,save_subdir=None):

    if save_subdir is None:
        save_subdir = 'state_corrs/'
        if not os.path.isdir(fdir+save_subdir):
            os.mkdir(fdir+save_subdir)

    if subdir is None:
        if is_tldm:        subdir = 'tldm_states/'
        else:              subdir = 'peps_states/'

    if corr_type in ['SS','spin']:
        try:
            spin_corrs  = load_obj(fdir+subdir,fstr+'_SScorrs')

        except(IOError):
            print 'no corrs: '+fdir+subdir+fstr+'_SScorrs.pkl'
            try:
                gammas  = np.load(fdir+subdir+fstr+'_gammas.npy' , allow_pickle=True)
                lambdas = np.load(fdir+subdir+fstr+'_lambdas.npy', allow_pickle=True)
                
                if is_tldm:    pepx_state = TLDM_GL.TLDM_GL(gammas,lambdas)
                else:          pepx_state = PEPX_GL.PEPX_GL(gammas,lambdas)
            
                if do_mp:            spin_corrs = meas.meas_spin_corr_all_MP(pepx_state)
                else:                spin_corrs = meas.meas_spin_corr_all(pepx_state)
                save_obj(spin_corrs,fdir+save_subdir,fstr+'_SScorrs')
            
            except(IOError):
                print 'no file: '+fdir+subdir+fstr+'_gammas.npy'
                raise(IOError)

        return spin_corrs


    elif corr_type in ['DD','dimer']:
        try:
            dimer_corrs = load_obj(fdir+subdir,fstr+'_DDcorrs')
        except(IOError):
            try:
                gammas  = np.load(fdir+subdir+fstr+'_gammas.npy' , allow_pickle=True)
                lambdas = np.load(fdir+subdir+fstr+'_lambdas.npy', allow_pickle=True)
                
                if is_tldm:    pepx_state = TLDM_GL.TLDM_GL(gammas,lambdas)
                else:          pepx_state = PEPX_GL.PEPX_GL(gammas,lambdas)

                if do_mp:            dimer_corrs = meas.meas_dimer_corr_idx(pepx_state)
                else:                dimer_corrs = meas.meas_dimer_corr_idx(pepx_state)
                save_obj(dimer_corrs,fdir+save_subdir,fstr+'_DDcorrs')

            except(IOError):
                print 'no file: '+fdir+subdir+fstr+'_gammas.npy'
                raise(IOError)
    
        return dimer_corrs


def get_nrg(fdir,fstr,trotterH,XMAX2=100):
    ''' recalculate energy instead of reading from file '''

    gammas  = np.load(fdir+'peps_states/'+fstr+'_gammas.npy' , allow_pickle=True)
    lambdas = np.load(fdir+'peps_states/'+fstr+'_lambdas.npy', allow_pickle=True)

    pepx_state = PEPX_GL.PEPX_GL(gammas,lambdas)

    gs_nrg = PEPX_GL.meas_obs(pepx_state,trotterH,XMAX=XMAX2)

    return gs_nrg



L1,L2 = (5,5)

fdir_SU = 'j1j2_tldm_data/'
dt_SU = 0.1

## thermal data
fdir_FU = 'j1j2_tldm_qrDL_data/'
# fdir_FU = 'j1j2_tldm_data_2site/'
# fdir_FU = 'j1j2_tldm_data_full/'
# fdir_FU = 'j1j2_tldm_data_full_DL/'
is_tldm = True
dt_FU = 0.1

# ## hz data
# fdir_FU = 'j1j2_peps_data/'
# is_tldm = False
# dt_FU = 0.1 # 0.01


hx,hz = [0,0]
Ds = [4] #[2,3,4,5]
hzs = [0] # [0. ,0.1, 0.25, 0.5, 0.75, 1.0]
XMAX2 = 100

is_SU = False #True
is_FU = True
do_dimer = False

J2s_SU = np.linspace(0.5,0.9,9)   # np.arange(16)*0.1
J2s_SU = [0.6,0.65,0.7,0.75]
J2s_FU = np.linspace(0.3,0.7,9)  # [0.7, 0.75]  #[0.6,0.65,0.7,0.75]
J2s_FU = [1.0]

# measure order params vs D, J2
if is_SU:

    fdir = fdir_SU

    dt = dt_SU
    XMAX = 100
    
    hxstr = '%04d'%int(100*hx)
    hzstr = '%04d'%int(100*hz)
    dtstr = '%04d'%int(1000*np.abs(dt))
    
    J2s =  J2s_SU  

    do_spin = True
    do_dimer = False

    for DMAX in Ds:

        for J2 in J2s:

            J2str = '%04d'%int(100*J2)
            fstr  = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s_SU'%(L1,L2,DMAX,XMAX2,dtstr,hxstr,hzstr,J2str)

            try:
                #### spin corrs ####
                spin_corrs = get_corrs(fdir,fstr,'SS',do_mp=True)
        
                ## structure factor value per site
                neels = meas.neel_order(spin_corrs,(1,1))/L1/L2
                # strip += [meas.neel_order(spin_corrs,(1,0))/L1/L2]
                strip_x = meas.neel_order(spin_corrs,(1,0))/L1/L2
                strip_y = meas.neel_order(spin_corrs,(0,1))/L1/L2
                print 'neel', neels, 'strip_x', strip_x, 'strip_y', strip_y 

                #### dimer corrs ####
                if do_dimer:
                    dimer_corrs = get_corrs(fdir,fstr,'DD')

                    ## structure factor value per site
                    cols  = meas.dimer_dimer_SF(spin_corrs,dimer_corrs,'col')
                    vbcs  = meas.dimer_dimer_SF(spin_corrs,dimer_corrs,'vbc')
        
                    print 'cols', cols, 'vbc', vbcs

            except(IOError):
                print 'no file: '+fdir+'peps_states/'+fstr+'_gammas.npy'
    


if is_FU:

    fdir = fdir_FU

    dt = dt_FU
    
    dtstr = '%04d'%int(1000*np.abs(dt))
    
    J2s = J2s_FU
    
    for DMAX in Ds:
    
        print 'looking at DMAX', DMAX
    
        for J2 in J2s:

            J2str = '%04d'%int(100*J2)

            for hz in hzs:

                hxstr = '%04d'%int(100*hx)
                hzstr = '%04d'%int(100*hz)

                if is_tldm:
                    temp_str  = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s_'%(L1,L2,DMAX,XMAX2,dtstr,hxstr,hzstr,J2str)
                    fn_gs = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'tldm_states/')], temp_str+'*_gammas.npy')
                    fn_ls = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'tldm_states/')], temp_str+'*_lambdas.npy')
                else:
                    temp_str  = 'L%dx%d_D%d_Xo%d_dt%s_hx%s_hz%s_J2%s_FU_'%(L1,L2,DMAX,XMAX2,dtstr,hxstr,hzstr,J2str)
                    fn_gs = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'peps_states/')], temp_str+'*_gammas.npy')
                    fn_ls = fnmatch.filter([fn for fn in os.listdir('./'+fdir+'peps_states/')], temp_str+'*_lambdas.npy')

                for fnum in range(len(fn_gs)):
                    
                    fstr = fn_gs[fnum][:-11]
                    print 'fstr', fstr
                    X_ind = fstr.find('X',-5)
                    Xstr = fstr[X_ind:]

                    try:
                        #### spin corrs ####
                        if is_tldm:   spin_corrs = get_corrs(fdir,fstr,'SS',do_mp=True,is_tldm=True,subdir='tldm_states/')
                        else:         spin_corrs = get_corrs(fdir,fstr,'SS',do_mp=True)
        
                        ## structure factor value per site

                        ## XY-Z symmetry
                        neels_xyz = np.array( meas.neel_order_XYZ(spin_corrs,(1,1)) )/L1/L2
                        strip_xyz_x = np.array( meas.neel_order_XYZ(spin_corrs,(1,0)) )/L1/L2
                        strip_xyz_y = np.array( meas.neel_order_XYZ(spin_corrs,(0,1)) )/L1/L2
                        use_x = np.sum(strip_xyz_x) > np.sum(strip_xyz_y)
                        if use_x:   strip_xyz = strip_xyz_x
                        else:       strip_xyz = strip_xyz_y
                        print 'neel xx+yy', np.sum(neels_xyz[:2]), 'neel zz', neels_xyz[-1]
                        print 'strip xx+yy', np.sum(strip_xyz[:2]), 'strip zz', strip_xyz[-1]

                        #### dimer corrs ####
                        if do_dimer:
                            if is_tldm:    dimer_corrs = get_corrs(fdir,fstr,'DD',do_mp=True,is_tldm=True,subdir='tldm_states')
                            else:          dimer_corrs = get_corrs(fdir,fstr,'DD',do_mp=True)

                            ## structure factor value per site
                            cols  = meas.dimer_dimer_SF(spin_corrs,dimer_corrs,'col')
                            vbcs  = meas.dimer_dimer_SF(spin_corrs,dimer_corrs,'vbc')
        
                            print 'cols', cols, 'vbc', vbcs

                    except(IOError):
                        if is_tldm:   print 'no file: '+fdir+'tldm_states/'+fstr+'_gammas.npy'
                        else:         print 'no file: '+fdir+'peps_states/'+fstr+'_gammas.npy'
    
