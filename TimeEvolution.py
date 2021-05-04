import numpy as np
from scipy import linalg
import copy
import time

import tens_fcts as tf
import MPX
import Liouv_MPX as LMPX



#:::::::::::::::::::::::::::::::::::::::::::::::::
# function to deal with different time evolutions
#::::::::::::::::::::::::::::::::::::::::::::::::::

def timeEvolve(mpx_, H_MPO, dt, t, te_method, order=4, compress=True, DMAX=100,
               direction=0, normalize=True, mps_type='mps', norm=1.0):
    """
    mpo is in the desired form according to te_method
    """

    nsteps = int(abs(t/dt)+1.0e-12)

    mpx = mpx_.copy()
    if te_method == 'exact':
        if type(H_MPO) is np.ndarray:
            exp_mpo = H_MPO
        else:
            exp_mpo = exact_exp(H_MPO,dt)
        mps_t  = exact_TE(mpx,exp_mpo,nsteps,direction,normalize,mps_type,norm,DMAX=DMAX)
        errs_t = [np.array([])]
    elif te_method == 'Taylor':
        # if isinstance(H_MPO,list):
        #     taylor_mpos = H_MPO
        # else:
        #     taylor_mpos = Taylor_MPO(H_MPO,dt,order)
        
        ## H_MPO:  list of mpos from Taylor_MPO(H_MPO,dt,order)
        mps_t, errs_t = Taylor_TE(mpx, H_MPO, dt, nsteps, order, compress, DMAX, direction,
                                  normalize, mps_type,norm)

    elif te_method == 'RK4':
        mps_t = []
        errs_t = []
        time = 0
        
        while time < abs(t):
            mps, e1 = RK4(mps, H_MPO, dt, compress, DMAX)
            mps_t.append(mps)
            errs_t.append(e1)

            time += abs(dt)
    else:
        raise ValueError, 'te_method should be exact, Taylor, or RK4'

    return mps_t, errs_t


#:::::::::::::::::::::::::::::::::::::::::::::::::
#             exact time evolution
#:::::::::::::::::::::::::::::::::::::::::::::::::

def exact_exp(mpo,dt):
    """
    returns exact exp(dt*H) (ndarray)

    parameters
    ----------
    mpo:   mpx object or Operator object
    dt:    time step
    """


    L = len(mpo)
    time_block = MPX.getSites(mpo,0,L)
    # print 'time_block'
    # for s in mpo:  print s.transpose(0,3,1,2), '\n'
    axT = tf.site2io(L)
    time_block = time_block.transpose(axT)

    t_sh = time_block.shape
    sqdim = int(np.sqrt(np.prod(t_sh)))
    H = time_block.reshape(sqdim,sqdim)
    
    if np.abs(np.real(dt)) < 1.0e-8: 
        U = linalg.expm(1.*np.imag(dt)*H)
    else:
        U = linalg.expm(-1j*dt*H)

    # U = linalg.expm(-1j*dt*H)

    evals = linalg.eigvals(U)

    U = U.reshape(t_sh)
    axT  = tf.io2site(L)
    expH = U.transpose(axT)

    return expH


def exact_expL(mpo,dt):

    # if np.abs(np.real(dt)) < 1.0e-8:   # imag time evolution
    #     expHf = exact_exp(mpo, dt)
    #     expHb  = PEPX.iden([ds[0] for ds in mpo.phys_bonds])
    # elif np.abs(np.imag(dt)) < 1.0e-8: # real time evolution
    #     expHf = exact_exp(mpo, dt)
    #     expHb = exact_exp(mpo,-dt)
    # else:                              # complex time evolution
    #     dt_ = -1*np.real(dt) + 1.j*np.imag(dt)

    expHf = exact_exp(mpo,dt)
    expHb = exact_exp(mpo,-1*np.conj(dt))

    L = len(mpo)
    axF = [0]
    axB = [1]
    for xx in range(L):
        axF += [xx*4+2,xx*4+4]
        axB += [xx*4+5,xx*4+3]
    axF += [2+L*4]
    axB += [2+L*4+1]
   
    expL = np.einsum(expHf,axF,expHb,axB)
    expL = tf.reshape(expL,'ii,'*(2*len(mpo)+1)+'ii')

        

    # print 'test expL'
    # 
    # if L==2:        expLT = np.conj(expL).transpose([0,2,1,4,3,-1])
    # elif L==4:      expLT = np.conj(expL).transpose([0,2,1,4,3,6,5,8,7,-1])
    # else:
    #     return expL

    # temp = MPX.dot_block_block( expLT, expL )
    # iden = MPX.eye([d[0]**2 for d in mpo.phys_bonds]).getSites()
    # print 'expLt exp-Lt - iden', np.linalg.norm(temp-iden)

    # exit()

    return expL


# @profile
def exact_TE(mps_,exact_exp,nSteps,direction=0,normalize=True,mps_type='mps',norm=1.0, DMAX=-1, chk_canon=True): #False):
    """
    performs exact time evolution
    """
   
    mps_t = []
    mps = mps_.copy()

    exp_mpo = tf.decompose_block(exact_exp,len(mps),0,-1,svd_str=3)
    for i in range(nSteps):
       # apply block to state, and then compress
       mps = MPX.dot_block(mps,0,len(mps),exact_exp,block_order='site',direction=(direction),DMAX=DMAX,
                           normalize=normalize,compress=True)

       # # dot, and then compress  
       # mps = MPX.dot(exp_mpo,mps)
       # mps = MPX.normalize(mps,1.)
       # mps, errs = MPX.compress_1(mps,-1,1)
       # mps, errs = MPX.compress(mps,DMAX,0)  
 
       # # dot_compress       
       # mps,errs = MPX.dot_compress(exp_mpo,mps,DMAX,(direction+1)%2)
       # mps = MPX.normalize(mps,1.)
       # mps, errs = MPX.compress_1(mps,-1,direction)

       # mps, errs = MPX.compress(full*(1./full_norm),DMAX,direction)

       # if normalize:
       #     if   mps_type == 'mps':   mps = MPX.normalize(mps,norm)
       #     elif mps_type == 'rho':   mps = LMPX.normalize(mps,norm)

       if chk_canon:   MPX.check_canon(mps,direction)

       mps_t.append(mps.copy())

    return mps_t


def expH_product_mpo(op_list,t):
    '''
    op_list is list of operators (dxd matrices)
    exponentiate ignoring interactions
    obtain something like exp(H) = exp(sum_i o_i) = prod_i exp(o_i)
    '''
    exp_s = []
    ds = []
    for s in op_list:
        d = s.shape[0]
        exp_s.append( linalg.expm((-1.j*t)*s).reshape(1,d,d,1) )
        ds.append((d,d))

    return MPX.MPX(exp_s,phys_bonds=ds)


def expL_product_mpo(op_list,t):
    '''
    make superoperator of the form exp(-1j*t*H) x exp(1j*t*H)
    '''
    expH  = expH_product_mpo(op_list, t)   # list of 1 x d x d x 1 matrices
    expH_ = expH_product_mpo(op_list,-t)

    expL = []
    for i in range(len(op_list)):
        expL.append( np.einsum('labr,ij->lajbir',expH[i],expH_[i][0,:,:,0]) )
        
    expL  = MPX.MPX(expL)
    return LMPX.combineBonds(expL)


#:::::::::::::::::::::::::::::::::::::::::::::::::
#          approx time evolution methods
#:::::::::::::::::::::::::::::::::::::::::::::::::


def Taylor_MPO(mpo, dt, order=4):
    """
    returns MPOs (H-a)(H-b) ... to be applied sequentially for time evolution

    exp(aH) = 1 + aH + 1/2 a^2 H^2 + 1/6 a^3 H^3 ... = x0(H-x1)(H-x2)(H-x3) ...
    
    obtains and applies (H-x) sequentially

    Parameters:
    -----------
    mpo:   MPX representation of H
    dt:    real-time time step
    order: approximation order
    """
  
    L = len(mpo)
    ds = [d[0] for d in mpo.phys_bonds]
    id_mpo = MPX.eye(ds)


    if order == 4:
        za = 0.270556 - 2.50478j
        zb = 0.270556 + 2.50478j
        zc = 1.72944  - 0.888974j
        zd = 1.72944  + 0.888974j

        zs = [za,zb,zc,zd]
        scaling = 1./(24**(0.25))
    elif order == 1:
        zs = [1.+0.j]
        scaling = 1.
    elif order == 2:
        zs = [1+1j,1-1j]
        scaling = 1./(2.**0.5)
    elif order ==3:
        zs = [1.59607, 0.701964-1.80734j, 0.701964+1.80734j]
        scaling = 1./(6**(1./3))    # 1/3!: normal scaling.   power ^(1./3) bc scaling applied to each MPO layer
    elif order == 0:
        return [id_mpo]
    elif order == 8:
        z1 = -2.03982 + 4.71861j
        z2 = -2.03982 - 4.71861j
        z3 = 0.788794 + 3.77181j
        z4 = 0.788794 - 3.77181j
        z5 = 2.28643  + 2.37771j
        z6 = 2.28643  - 2.37771j
        z7 = 2.9646   - 0.808878j
        z8 = 2.9646   + 0.808878j
 
        zs = [z1,z2,z3,z4,z5,z6,z7,z8]
        scaling = 1./(40320.**(1./8))
    else:  
        raise ValueError, 'Taylor expansion of order %d not yet implemented'%order


    mpo_list = []
    for z in zs:
        offset = MPX.mul(z,id_mpo)
        # mpo_eff, errs = MPX.compress(MPX.mul(scaling, MPX.mul(-1j*dt, mpo) - offset),-1)
        mpo_eff = MPX.mul(scaling, MPX.mul(-1j*dt, mpo) + offset)
        mpo_list.append(mpo_eff)

    # for z in zs:
    #     offset = MPX.mul(z,id_mpo)
    #     mpo_eff, errs = MPX.compress(MPX.mul(scaling, MPX.mul(-1j*dt, mpo) - offset),-1)
    #     # print 'mpo_eff', [s.shape[0] for s in mpo_eff]
    #     try:
    #         mpo_f = MPX.compress ( MPX.dot( mpo_eff, mpo_f ), 100)[0]
    #     except:  # mpo not defined
    #         mpo_f = mpo_eff.copy()

    # mpo_list = [mpo_f]
    # print 'mpo_final', [s.shape[0] for s in mpo_f]

    return mpo_list


# @profile
def Taylor_TE(mps_, mpo_list, dt, nSteps, order=4, compress=True, DMAX=100, direction=0,
              normalize=True, mps_type='mps',norm=1.0):
    """
    perform time evolution using Taylor approximation
    
    Parameters:
    ---------- 
  
    mps:          mpx object to do time evolution on
    taylor_mpos:  list of mpos to apply to mps (already includes dt info)
    nSteps:       number of time steps
    dot_fn:       MPX.dot (tMPS) or MPX.dot_compress (tDMRG)
    """

    step = 0
    errs_t = []
    mps_t  = []

    time1 = time.time()
    # taylor_mpos = Taylor_MPO(H_MPO,dt,order)
    taylor_mpos = mpo_list
    # print 'time to get mpos', time.time()-time1

    mps = mps_.copy()
    errs = np.array([0.]*(len(mps)-1))
    dir_ = direction
    if compress:    # compress as one dots
        while step < nSteps:
            # print 'TE step', step

            time1 = time.time()
            mps_o = mps.copy()
            for mpo in taylor_mpos:
                mps, e = mpo.dot_compress(mps, DMAX, dir_)
                # for s in mps:  print 'mps after c1', s.shape
                errs += e
                dir_  = (dir_+1)%2

            if normalize:
                ## overlap with initial state, should be about 1-dt for small dt
                errs = [1-MPX.vdot(mps_o,mps)]
                # if np.abs(errs) > 1:
                #     print 'TE timestep: ', step,', overlap:', errs
                #     continue
    
                if   mps_type == 'mps':   mps = MPX.normalize(mps,norm)
                elif mps_type == 'rho':   mps = LMPX.normalize(mps,norm)


            errs_t.append(errs[:])
            mps_t.append(mps.copy())
            step += 1
            # print '1 time step', time.time()-time1, [s.shape[0] for s in mps_t[0]]
    else:           # do tMPS (compress after time step)
        while step < nSteps:
            # print 'TE step', step

            time1 = time.time()
            mps_o = mps.copy()
            for mpo in taylor_mpos:
                temp = MPX.compress(MPX.dot(mpo,mps), -1, (dir_+1)%2)[0]
                mps, e = MPX.compress( temp, DMAX, dir_ )
                # mps, e = MPX.compress(mpo.dot(mps), DMAX, dir_)

                # for s in mps:  print 'mps after c2', s.shape
                # print 'taylor TE', [s.shape[0] for s in mps]
                # mps = MPX.compress(mps,DMAX)[0]
                # print 'taylor te', [s.shape[0] for s in mps]
                errs += e
                dir_  = (dir_+1)%2

            # mps, e = MPX.compress(mps, DMAX, dir_)

            if normalize:
                errs = [1-MPX.vdot(mps_o,mps)]
                # print 'TE timestep overlap', errs
                if   mps_type == 'mps':   mps = MPX.normalize(mps,norm)
                elif mps_type == 'rho':   mps = LMPX.normalize(mps,norm)

            errs_t.append(errs[:])
            mps_t.append(mps.copy())
            step += 1
            # print '1 time step', time.time()-time1, [s.shape[0] for s in mps_t[0]]

    return mps_t, errs_t



def trotterTE(mps, H_, tau, nSteps, DMAX=100, normalize=True, te_type='mps'):

    L = len(mps)
   
    time1 = time.time()

    psi = mps.copy()
    mps_ts = []
    
    mpo = H_.ops   
    ns  = H_.ns

    if te_type in ['mpo','rho']:    # mps assumed to be rho --> flattened
        exp_blk = exact_expL( mpo, tau )            # trotter step
        try:
            exp_blkL = exact_expL( H_.opsL, tau )   # trotter step for (L-1,L) bond
        except:
            exp_blkL = exp_blk

    else:
        exp_blk = exact_exp( mpo, tau )            # trotter step
        try:
            exp_blkL = exact_exp( H_.opsL, tau )   # trotter step for (L-1,L) bond
        except:
            exp_blkL = exp_blk
    

    temp, s_list = tf.decompose_block(exp_blkL,ns,0,-1,'ijk,...',return_s=True)
    exp_mpoL = MPX.MPX(temp)  # MPX.split_singular_vals(temp,s_list)

    temp, s_list = tf.decompose_block(exp_blk,ns,0,-1,'ijk,...',return_s=True)
    exp_mpo = MPX.MPX(temp)   # MPX.split_singular_vals(temp,s_list)

    edge_order = range(len(H_.edges))
    # edge_order = range(0,len(H_.edges),2) + range(1,len(H_.edges),2)
     

    i = 0
    while i < nSteps:
    
        for step in edge_order:
       
            ind1, ind2 = H_.edges[step]
            apply_swaps = [x for x in range(ind2-1,ind1,-1)]
            try:
               mpo_scale = H_.weights[step]
            except(TypeError):
               mpo_scale = H_.weights
        
            # # apply swap gates s.t. mpo operating on n.n.a and then do trotter step
            # for sw in apply_swaps:
            #     psi[sw:sw+2] = MPX.dot_compress( swap_gate, psi[sw:sw+2], -1 )[0]
            
            # put in correct canonical form  
            if ind1 == 0:      # all right normalized
                psi = MPX.compress( psi, -1, direction=1)[0]
                # print '0 norm', MPX.norm(psi,direction=1)
                MPX.check_canon(psi,1)
            elif ind1 == L-2:  # all left normalized
                psi = MPX.compress( psi, -1, direction=0)[0]
                # print 'L-2 norm', MPX.norm(psi)
                MPX.check_canon(psi,0)
            else:
                psi[:ind1+1] = MPX.compress( psi[:ind1+1], -1, direction=0)[0]   # L canon up to ind+1 (through ind)
                MPX.check_canon(psi,0,ind1)
                psi[ind1:] = MPX.compress( psi[ind1:], -1, direction=1)[0]       # R canon up to ind (through ind+1)
                MPX.check_canon(psi,1,ind1+1)

            # if ind1 == L-2:
            #     psi_ite = MPX.dot_block( psi, ind1, ns, exp_mpoL, block_order='site',DMAX=DMAX )
            # else:
            #     psi_ite = MPX.dot_block( psi, ind1, ns, exp_mpo, block_order='site', DMAX=DMAX )

            psi_ite = psi.copy()
            if ind1 == L-2:
                # ## block TE
                # psi_ite = MPX.dot_block(psi,ind1,ns,exp_blkL,block_order='site',DMAX=DMAX,normalize=True)

                ## block TE on site
                psi_op = MPX.dot_block(psi[ind1:ind1+ns],0,ns,exp_blkL,block_order='site',DMAX=DMAX,normalize=True)
                psi_ite[ind1:ind1+ns] = psi_op

                # ## dot_compress
                # psi_op, errs = MPX.dot_compress( exp_mpoL,psi[ind1:ind1+ns], DMAX=DMAX )
                # psi_norm = MPX.norm(psi_op)
                # psi_ite[ind1:ind1+ns] = psi_op*(1./psi_norm)

                # ## dot + compress
                # psi_op = MPX.dot( exp_mpoL, psi[ind1:ind1+ns])
                # psi_norm = MPX.norm(psi_op)
                # psi_op, errs = MPX.compress( psi_op*(1./psi_norm), DMAX)  
                # psi_ite[ind1:ind1+ns] = psi_op
            else:
                # # ## block TE
                # psi_ite = MPX.dot_block(psi,ind1,ns,exp_blk,block_order='site',DMAX=DMAX,normalize=True)

                ## block TE on site
                psi_op = MPX.dot_block(psi[ind1:ind1+ns],0,ns,exp_blkL,block_order='site',DMAX=DMAX,normalize=True)
                temp = MPX.dot_block(psi[ind1:ind1+ns],0,ns,exp_blkL,block_order='site',DMAX=DMAX,
                                     normalize=True,compress=False)
                psi_op = MPX.MPX(tf.decompose_block(temp,ns,0,DMAX,3))
                psi_ite[ind1:ind1+ns] = psi_op

                # ## dot_compress
                # psi_op, errs = MPX.dot_compress( exp_mpo, psi[ind1:ind1+ns], DMAX=DMAX )
                # psi_norm = MPX.norm(psi_op)
		# psi_ite[ind1:ind1+ns] = psi_op*(1./psi_norm)

                ## dot + compress
                psi_op = MPX.dot( exp_mpo, psi[ind1:ind1+ns])
                psi_norm = MPX.norm(psi_op)
                psi_op, errs = MPX.compress( MPX.mul(1./psi_norm,psi_op), DMAX, direction=0, ref_bl = temp)
                psi_ite[ind1:ind1+ns] = psi_op

            psi = psi_ite.copy()
        
            # for sw in apply_swaps[::-1]:
            #     psi[sw:sw+1] = MPX.dot_compress( swap_gate, psi[sw:sw+1], -1 )[0]

        if normalize:
            if te_type in ['mpo','rho']: 
                psi = LMPX.normalize(psi)
            else:
                psi = MPX.normalize(psi)
        # psi = MPX.compress(psi,-1,direction=1)[0]   # all right normalized

        mps_ts.append(psi.copy())
    
        i += 1

    return mps_ts


###### not yet fully implemented / debugged!! ######
def tDMRG(mps, H, dt, totT, gse=None, num_sites=2, isH=True, DMAX=100, mps_type='mps'):
    """
    Ronca et al. 2017
    perform time evolution by calculating via sweep algorithm

    |1> = P[k](t+dt) dt (H(t)-e0) P[k](t) |x[k](t)>
    |2> = P[k](t+dt) dt (H(t+dt/2)-e0) P[k](t+dt) [|x[k](t)> + 1/2|1[k]>]
    |3> = P[k](t+dt) dt (H(t+dt/2)-e0) P[k](t+dt) [|x[k](t)> + 1/2|2[k]>]
    |4> = P[k](t+dt) dt (H(t+dt/2)-e0) P[k](t+dt) [|x[k](t)> + |3>]

    |x(t+dt)> = 1/6(|1> + 2|2> + 2|3> + |4>)

    where [k] indicates site 
    P[k](t) projects onto renormalized basis of |x(t)> at site k
    transformed to next site using density matrices of |x[k](t)> and |x[k](t+dt)> 

    note:  mpo input is already H-e0

    """

    import MPX_tDMRG 
 
    step = 0
    nSteps = int(totT/dt)

    errs_t = []
    mps_t  = []

    time1 = time.time()

    errs = np.array([0.]*(len(mps)-1))
   
    if gse is None:
        import MPX_GSEsolver
        gs_evec, gse = MPX_GSEsolver.solver(self.QL,target=0,DMAX=20, conv_err=1.0e-3,isH=isH)
   
    if np.abs(gse) < 1.0e-3:   H_ = MPX.mul(dt,H)
    else:                      H_ = MPX.mul(dt,H + MPX.eye([d[0] for d in H.phys_bonds])*(-gse))
    # if np.abs(gse) < 1.0e-3:   H_ = H
    # else:                      H_ = H + MPX.eye([d[0] for d in H.phys_bonds])*(-gse)
    mps_ = mps.copy()


    while step < nSteps:

        mps_ = MPX_tDMRG.solver_dt( mps, H_, dt, num_sites, DMAX)[0]

        if normalize:
            ## overlap with initial state, should be about 1-dt for small dt
            if   mps_type == 'mps':   mps_ = MPX.normalize(mps_,norm)
            elif mps_type == 'rho':   mps_ = LMPX.normalize(mps_,norm)

        mps_t.append(mps_.copy())
        step += 1

    return mps_t, errs_t


