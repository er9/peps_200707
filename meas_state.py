import time
import multiprocessing as mp
from multiprocessing import queues
import numpy as np

import PEPX
import PEPX_GL
import TLDM_GL



# measure <S.S> correlation between two sites at idx0, idx1
def meas_spin_corr(state,idx0,idx1,XMAX=100,envs=None,contract_SL=False,return_inds=False):

    if isinstance(state,TLDM_GL.TLDM_GL):     meas_fn = TLDM_GL.meas_obs
    elif isinstance(state,PEPX_GL.PEPX_GL):   meas_fn = PEPX_GL.meas_obs
    elif isinstance(state,PEPX.PEPX):         meas_fn = PEPX.meas_obs
    else:
        print type(state)
        raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'

    sxsx = meas_fn(state, [['SX','SX'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)
    sysy = meas_fn(state, [['SY','SY'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)
    szsz = meas_fn(state, [['SZ','SZ'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)

    if return_inds:    return idx0,idx1,(sxsx,sysy,szsz)
    else:              return (sxsx,sysy,szsz)


# measure <S.S> correlation between two sites at idx0, idx1
def meas_spin_corr_MP(state,idx0,idx1,XMAX=100,envs=None,contract_SL=False,result_list=[]):

    if isinstance(state,TLDM_GL.TLDM_GL):     meas_fn = TLDM_GL.meas_obs
    elif isinstance(state,PEPX_GL.PEPX_GL):   meas_fn = PEPX_GL.meas_obs
    elif isinstance(state,PEPX.PEPX):         meas_fn = PEPX.meas_obs
    else:
        print type(state)
        raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'

    sxsx = meas_fn(state, [['SX','SX'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)
    sysy = meas_fn(state, [['SY','SY'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)
    szsz = meas_fn(state, [['SZ','SZ'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)

    result_list += [(idx0,idx1,(sxsx,sysy,szsz))]
    return


# measure <S.S> correlation between two sites at idx0, idx1
def meas_spin_corr_MP2(state,idx0,idx1,XMAX=100,envs=None,contract_SL=False,out_q=None):

    if isinstance(state,TLDM_GL.TLDM_GL):     meas_fn = TLDM_GL.meas_obs
    elif isinstance(state,PEPX_GL.PEPX_GL):   meas_fn = PEPX_GL.meas_obs
    elif isinstance(state,PEPX.PEPX):         meas_fn = PEPX.meas_obs
    else:
        print type(state)
        raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'

    sxsx = meas_fn(state, [['SX','SX'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)
    sysy = meas_fn(state, [['SY','SY'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)
    szsz = meas_fn(state, [['SZ','SZ'], [idx0,idx1]],XMAX=XMAX,envs=envs,contract_SL=contract_SL)

    if out_q is not None:   out_q.put( (idx0,idx1,(sxsx,sysy,szsz)) )
    return


# measure <S.S> correlation between all pair of sites in lattice
def meas_spin_corr_all(state,XMAX=100,contract_SL=False):

    print('serial meas spin corr all')

    L1,L2 = state.shape
   
    if   isinstance(state,TLDM_GL.TLDM_GL):    bnds_fn = TLDM_GL.meas_get_bounds
    elif isinstance(state,PEPX_GL.PEPX_GL):    bnds_fn = PEPX_GL.meas_get_bounds
    elif isinstance(state,PEPX.PEPX):          bnds_fn = PEPX.meas_get_bounds
    else:
        print type(state)
        raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'
    
    bounds = bnds_fn(state,None,(0,0),XMAX,contract_SL)

    meas_all = {}

    for i0,j0 in np.ndindex((L1,L2)):

        ind0 = i0*L2 + j0   # linear index
        meas_all[(i0,j0)] = {}

        for i1,j1 in np.ndindex((L1,L2)):

            ind1 = i1*L2 + j1
            if ind1 < ind0:   continue

            print('measuring (%d,%d), (%d,%d)'%(i0,j0,i1,j1))

            s2 = meas_spin_corr(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL)
            meas_all[(i0,j0)][(i1,j1)] = s2

    return meas_all


#### Pool method doesn't work -- doesn't wait for function to finish running ###
#### works for fast functions  ###

# # measure <S.S> correlation between all pair of sites in lattice
# def meas_spin_corr_all_MP(state,XMAX=100,contract_SL=False):
# 
#     print 'meas spin corr all MP'
# 
#     L1,L2 = state.shape
#    
#     if   isinstance(state,TLDM_GL.TLDM_GL):    bnds_fn = TLDM_GL.meas_get_bounds
#     elif isinstance(state,PEPX_GL.PEPX_GL):    bnds_fn = PEPX_GL.meas_get_bounds
#     elif isinstance(state,PEPX.PEPX):          bnds_fn = PEPX.meas_get_bounds
#     else:
#         print type(state)
#         raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'
#     
#     bounds = bnds_fn(state,None,(0,0),XMAX,contract_SL)
# 
#     meas_all = {}
# 
#     ## multiprocessing part
#     pool = mp.Pool(mp.cpu_count())
#     print 'num cpu', mp.cpu_count()
# 
#     def collect_result(result):
#         print 'result', result
#         global meas_all
#         idx0,idx1,obs_val = result
#         meas_all[idx0][idx1] = obs_val
# 
# 
#     print type(state), type(state.lambdas)
# 
#     res_obj = []
# 
#     for i0,j0 in np.ndindex((2,2)):  #((L1,L2)):
# 
#         ind0 = i0*L2 + j0   # linear index
#         meas_all[(i0,j0)] = {}
# 
#         for i1,j1 in np.ndindex((L1,L2)):
# 
#             ind1 = i1*L2 + j1
#             if ind1 < ind0:   continue
# 
#             # s2 = meas_spin_corr(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL)
#             res_obj += [pool.apply_async( meas_spin_corr, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,True))]
#             # res_obj += [pool.apply_async( toy_f, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,True))]
#             # print type(res_obj[-1].get())
#             
#             # pool.apply_async( meas_spin_corr, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,True),
#             #                   callback = collect_result )
#             # meas_all[(i0,j0)][(i1,j1)] = s2
# 
#     pool.close()
#     pool.join()
# 
#     # for ro in res_obj:
#     #     ro.wait()
#     #     if ro.ready():
#     #         print ro.ready()
#     #         print ro.get()
#     #         idx0,idx1,obs_val = ro.get()
#     #         meas_all[idx0][idx1] = obs_val
# 
# 
#     return meas_all

# #### results list doesn't work on pauling (can't access) ####
# # measure <S.S> correlation between all pair of sites in lattice
# def meas_spin_corr_all_MP(state,XMAX=100,contract_SL=False):
# 
#     print 'meas spin corr all MP'
# 
#     L1,L2 = state.shape
#    
#     if   isinstance(state,TLDM_GL.TLDM_GL):    bnds_fn = TLDM_GL.meas_get_bounds
#     elif isinstance(state,PEPX_GL.PEPX_GL):    bnds_fn = PEPX_GL.meas_get_bounds
#     elif isinstance(state,PEPX.PEPX):          bnds_fn = PEPX.meas_get_bounds
#     else:
#         print type(state)
#         raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'
#     
#     bounds = bnds_fn(state,None,(0,0),XMAX,contract_SL)
# 
#     manager = mp.Manager()
# 
#     result_list = manager.list()
#     meas_all = {}
# 
#     ## multiprocessing part
#     # pool = mp.Pool(mp.cpu_count())
#     # print 'num cpu', mp.cpu_count()
# 
#     ps = []
# 
#     for i0,j0 in np.ndindex((L1,L2)):
# 
#         ind0 = i0*L2 + j0   # linear index
#         meas_all[(i0,j0)] = {}
# 
#         for i1,j1 in np.ndindex((L1,L2)):
# 
#             ind1 = i1*L2 + j1
#             if ind1 < ind0:   continue
# 
#             # s2 = meas_spin_corr(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL)
#             p = mp.Process( target=meas_spin_corr_MP, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,result_list))
#             p.start()
#             ps += [p]
#             # res_obj += [pool.apply_async( toy_f, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,True))]
#             # print type(res_obj[-1].get())
#             
#             # pool.apply_async( meas_spin_corr, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,True),
#             #                   callback = collect_result )
#             # meas_all[(i0,j0)][(i1,j1)] = s2
# 
#     for p in ps:
#         p.join()
# 
#     # print result_list
#     for ro in result_list:
#         idx0,idx1,obs_val = ro
#         meas_all[idx0][idx1] = obs_val
# 
#     return meas_all


#### results list doesn't work on pauling (can't access) ####
# measure <S.S> correlation between all pair of sites in lattice
def meas_spin_corr_all_MP_v0(state,XMAX=100,contract_SL=False):

    print 'meas spin corr all MP'
    print 'num cpu', mp.cpu_count()

    L1,L2 = state.shape
   
    if   isinstance(state,TLDM_GL.TLDM_GL):    bnds_fn = TLDM_GL.meas_get_bounds
    elif isinstance(state,PEPX_GL.PEPX_GL):    bnds_fn = PEPX_GL.meas_get_bounds
    elif isinstance(state,PEPX.PEPX):          bnds_fn = PEPX.meas_get_bounds
    else:
        print type(state)
        raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'
    
    bounds = bnds_fn(state,None,(0,0),XMAX,contract_SL)

    out_q = queues.SimpleQueue()
    meas_all = {}

    ## multiprocessing part
    # pool = mp.Pool(mp.cpu_count())
    # print 'num cpu', mp.cpu_count()

    ps = []
    for i0,j0 in np.ndindex((L1,L2)):

        ind0 = i0*L2 + j0   # linear index
        meas_all[(i0,j0)] = {}

        for i1,j1 in np.ndindex((L1,L2)):

            ind1 = i1*L2 + j1
            if ind1 < ind0:   continue

            # s2 = meas_spin_corr(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL)
            p = mp.Process( target=meas_spin_corr_MP2, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,out_q))
            ps += [p]
            p.start()

    ## moving results into dict
    it = 0
    i = 0
    while len(ps) > 0: # and it < 1000:
        print 'in while'
        ps_ = []
        for p in ps:
            if p.is_alive():   ps_ += [p]
        ps = ps_
        #print 'here'
        time.sleep(10)
        # print 'here'

        while not out_q.empty():
            idx0,idx1,obs_val = out_q.get()
            meas_all[idx0][idx1] = obs_val
            i = i+1
            # print idx0,idx1,obs_val

        print 'iteration', it, len(ps), i
        it += 1

    print 'out while', it, len(ps)


    print 'empty q', i

    alive_p = 0
    for p in ps:
        if p.is_alive():    alive_p += 1

    print 'num alive ps, q size', alive_p, i
    if alive_p > 0:
        raise(RuntimeError),'still alive processes'

    for p in ps:
        p.terminate()

    return meas_all


#### results list doesn't work on pauling (can't access) ####
# measure <S.S> correlation between all pair of sites in lattice
# limit number of processes that run at the same time
def meas_spin_corr_all_MP(state,XMAX=100,contract_SL=False):

    print 'meas spin corr all MP'
    print 'num cpu', mp.cpu_count()

    L1,L2 = state.shape
   
    if   isinstance(state,TLDM_GL.TLDM_GL):    bnds_fn = TLDM_GL.meas_get_bounds
    elif isinstance(state,PEPX_GL.PEPX_GL):    bnds_fn = PEPX_GL.meas_get_bounds
    elif isinstance(state,PEPX.PEPX):          bnds_fn = PEPX.meas_get_bounds
    else:
        print type(state)
        raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'
    
    bounds = bnds_fn(state,None,(0,0),XMAX,contract_SL)

    out_q = queues.SimpleQueue()
    meas_all = {}

    ## multiprocessing part
    # pool = mp.Pool(mp.cpu_count())
    # print 'num cpu', mp.cpu_count()

    ps = []
    for i0,j0 in np.ndindex((L1,L2)):

        ind0 = i0*L2 + j0   # linear index
        meas_all[(i0,j0)] = {}

        for i1,j1 in np.ndindex((L1,L2)):

            ind1 = i1*L2 + j1
            if ind1 < ind0:   continue

            # s2 = meas_spin_corr(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL)
            p = mp.Process( target=meas_spin_corr_MP2, args=(state,(i0,j0),(i1,j1),XMAX,bounds,contract_SL,out_q))
            ps += [p]
            # p.start()


    ## moving results into dict
    running_ps = []
    remain_ps = ps

    it = 0
    i = 0
    while len(ps) > 0: # and it < 1000:
        print 'in while'

        for p in remain_ps:
            if len(running_ps) < 4:
                p.start()
                remain_ps.remove(p)
                running_ps.append(p)

        for p in running_ps:
            if not p.is_alive():   running_ps.remove(p)

        #print 'here'
        time.sleep(10)
        # print 'here'

        while not out_q.empty():
            idx0,idx1,obs_val = out_q.get()
            meas_all[idx0][idx1] = obs_val
            i = i+1
            # print idx0,idx1,obs_val

        print 'iteration', it, len(ps), i
        it += 1

    print 'out while', it, len(ps)


    print 'empty q', i

    alive_p = 0
    for p in ps:
        if p.is_alive():    alive_p += 1

    print 'num alive ps, q size', alive_p, i
    if alive_p > 0:
        raise(RuntimeError),'still alive processes'

    for p in ps:
        p.terminate()

    return meas_all


# measure correlations with all other dimers with dimer at idx0 (h or v)
def meas_dimer_corr_idx(state,idx0=None,XMAX=100,contract_SL=False):

    L1,L2 = state.shape

    if idx0 is None:   idx0 = ((L1-1)/2,(L2-1)/2)
    i0,j0 = idx0

    if isinstance(state,TLDM_GL.TLDM_GL):     meas_fn = TLDM_GL.meas_obs
    elif isinstance(state,PEPX_GL.PEPX_GL):   meas_fn = PEPX_GL.meas_obs
    elif isinstance(state,PEPX.PEPX):         meas_fn = PEPX.meas_obs
    else:
        print type(state)
        raise(TypeError),'meas spin corr:  state should be TLDM_GL, PEPX_GL or PEPX'

    # keep track of all indices and the meas vals
    meas_all = {}

    # assume horizontal dimer? 
    # choose neighbor with largest correlation?
    corr_v = meas_spin_corr(state,(i0,j0),(i0+1,j0),XMAX,contract_SL)
    corr_h = meas_spin_corr(state,(i0,j0),(i0,j0+1),XMAX,contract_SL)

    if np.abs(corr_v) > np.abs(corr_h):
        a0, a1 = (i0,j0),(i0+1,j0)
    else:
        a0, a1 = (i0,j0),(i0,j0+1)


    meas_idx = {}

    # vertical dimers
    for i1,j1 in np.ndindex((L1-1,L2)):

        # idx1 = i1*L2 + j1
        # if idx1 < idx0:   continue

        b0 = (i1,j1)
        b1 = (i1+1,j1)

        xxxx = meas_fn(state, [['SX','SX','SX','SX'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        yyxx = meas_fn(state, [['SY','SY','SX','SX'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        zzxx = meas_fn(state, [['SZ','SZ','SX','SX'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        xxyy = meas_fn(state, [['SX','SX','SY','SY'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        yyyy = meas_fn(state, [['SY','SY','SY','SY'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        zzyy = meas_fn(state, [['SZ','SZ','SY','SY'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        xxzz = meas_fn(state, [['SX','SX','SZ','SZ'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        yyzz = meas_fn(state, [['SY','SY','SZ','SZ'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        zzzz = meas_fn(state, [['SZ','SZ','SZ','SZ'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)

        s0xs1 = xxxx+xxyy+xxzz + yyxx+yyyy+yyzz + zzxx+zzyy+zzzz

        # meas_idx[(b0,b1)] = s0xs1
        meas_idx[(b0,b1)] = (xxxx,xxyy,xxzz,yyxx,yyyyy,yyzz,zzxx,zzyy,zzzz)

    # horizontal dimers
    for i1,j1 in np.ndindex((L1,L2-1)):

        # idx1 = i1*L2 + j1
        # if idx1 < idx0:   continue

        b0 = (i1,j1)
        b1 = (i1,j1+1)

        xxxx = meas_fn(state, [['SX','SX','SX','SX'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        yyxx = meas_fn(state, [['SY','SY','SX','SX'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        zzxx = meas_fn(state, [['SZ','SZ','SX','SX'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        xxyy = meas_fn(state, [['SX','SX','SY','SY'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        yyyy = meas_fn(state, [['SY','SY','SY','SY'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        zzyy = meas_fn(state, [['SZ','SZ','SY','SY'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        xxzz = meas_fn(state, [['SX','SX','SZ','SZ'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        yyzz = meas_fn(state, [['SY','SY','SZ','SZ'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)
        zzzz = meas_fn(state, [['SZ','SZ','SZ','SZ'],[a0,a1,b0,b1]], XMAX=XMAX, contract_SL=contract_SL)

        s0xs1 = xxxx+xxyy+xxzz + yyxx+yyyy+yyzz + zzxx+zzyy+zzzz

        # meas_idx[(b0,b1)] = s0xs1
        meas_idx[(b0,b1)] = (xxxx,xxyy,xxzz,yyxx,yyyy,yyzz,zzxx,zzyy,zzzz)

    meas_all = {(a0,a1): meas_idx}

    return meas_all



def neel_order_XYZ(corrs_data,k=(1,1)):   # k is scaled by pi

    tot_sum_xx, tot_sum_yy, tot_sum_zz = 0,0,0

    for idx0 in corrs_data.keys():
        for idx1 in corrs_data[idx0].keys():

            obs_val = corrs_data[idx0][idx1]    ## (xx,yy,zz)
            if isinstance(obs_val,np.float):
                raise(TypeError),'dictionary outdated--contains S2 but need XX,YY,ZZ info'

            rij = np.array(idx0)-np.array(idx1)
            coeff = np.exp( -1.j*np.pi*np.dot(k,rij))
            tot_sum_xx += coeff*obs_val[0]
            tot_sum_yy += coeff*obs_val[1]
            tot_sum_zz += coeff*obs_val[2]

            if idx0 != idx1:    # add the contriubtion from idx1-idx0
                rji = np.array(idx1)-np.array(idx0)
                coeff = np.exp( -1.j*np.pi*np.dot(k,rij))
                tot_sum_xx += coeff*obs_val[0]
                tot_sum_yy += coeff*obs_val[1]
                tot_sum_zz += coeff*obs_val[2]

    tot_xx = tot_sum_xx/len(corrs_data.keys())   # normalization, = L**2
    tot_yy = tot_sum_yy/len(corrs_data.keys())
    tot_zz = tot_sum_zz/len(corrs_data.keys())

    return tot_xx, tot_yy, tot_zz
    


def neel_order(corrs_data,k=(1,1)):   # k is scaled by pi

    tot_sum = 0

    for idx0 in corrs_data.keys():
        for idx1 in corrs_data[idx0].keys():

            obs_val = corrs_data[idx0][idx1]    ## (xx,yy,zz)
            if isinstance(obs_val,tuple):         obs_val = np.sum(obs_val)
            elif isinstance(obs_val,np.float):
                raise(DeprecationWarning),'dictionary outdated--contains S2 not XX,YY,ZZ info'

            rij = np.array(idx0)-np.array(idx1)
            coeff = np.exp( -1.j*np.pi*np.dot(k,rij))
            tot_sum += coeff*obs_val

            if idx0 != idx1:    # add the contriubtion from idx1-idx0
                rji = np.array(idx1)-np.array(idx0)
                coeff = np.exp( -1.j*np.pi*np.dot(k,rij))
                tot_sum += coeff*obs_val

    return tot_sum/len(corrs_data.keys())   # normalization, = L**2
    


def dimer_dimer_SF(corrs_data,dimer_data,phase,idx0=None):

    tot_sum = 0
    norm = 0

    i0,i1 = dimer_data.keys()[0]     # (i0,i1) : indices of central dimer
    orientation = np.array(i1)-np.array(i0)   # (0,1) for h, (1,0) for v
    if np.all(orientation == (0,1)):     is_hbond = False
    else:                                is_hbond = True

    if phase == 'col':      # aligned with idx0 dimer bond

        for j0,j1 in dimer_data[(i0,i1)].keys():

            obs_val = dimer_data[(i0,i1)][(j0,j1)]
            if isinstance(dict_val,tuple):         obs_val = np.sum(obs_val)
            elif isinstance(dict_val,np.float):
                raise(DeprecationWarning),'dictionary outdated--contains S2 not XX,YY,ZZ info'

            if j0 in [i0, i1]:   continue  # touching i0-i1  --> ignore
            if j1 in [i0, i1]:   continue
            
            # find <s.s> for i0-i1, j0-j1
            try:
                corri = corrs_data[i0][i1]
            except(KeyError):
                corri = corrs_data[i1][i0]
            
            try:
                corrj = corrs_data[j0][j1]
            except(KeyError):
                corrj = corrs_data[j1][j0]

            dimer_corr_val = obs_val - corri*corrj


            diff1 = np.array(j1)-np.array(j0)
            if np.all(np.abs(diff1) == orientation):
                if is_hbond:
                    if (j0[1]-i0[1])%2 == 0:       tot_sum += dimer_corr_val   # even num cols away from i0-i1
                    else:                          tot_sum -= dimer_corr_val   # odd  num cols away from i0-i1a
                else:   # is v_bond
                    if (j0[0]-i0[0])%2 == 0:       tot_sum += dimer_corr_val   # even num cols away from i0-i1
                    else:                          tot_sum -= dimer_corr_val   # odd  num cols away from i0-i1

                norm += 1

    elif phase ==  'vbc':   # plaquettes

        for j0,j1 in dimer_data[(i0,i1)].keys():

            obs_val = dimer_data[(i0,i1)][(j0,j1)]
            if isinstance(dict_val,tuple):         obs_val = np.sum(obs_val)
            elif isinstance(dict_val,np.float):
                raise(DeprecationWarning),'dictionary outdated--contains S2 not XX,YY,ZZ info'

            if j0 in [i0, i1]:   continue  # touching i0-i1  --> ignore
            if j1 in [i0, i1]:   continue
            
            # find <s.s> for i0-i1, j0-j1
            try:
                corri = corrs_data[i0][i1]
            except(KeyError):
                corri = corrs_data[i1][i0]
            
            try:
                corrj = corrs_data[j0][j1]
            except(KeyError):
                corrj = corrs_data[j1][j0]

            dimer_corr_val = obs_val - corri*corrj


            diff1 = np.array(j1)-np.array(j0)
            if np.all(np.abs(diff1) == orientation):    tot_sum += dimer_corr_val   # same bond orientation
            else:                                       tot_sum -= dimer_corr_val   # perpendicular bond orientation

            norm += 1

    return tot_sum/norm
