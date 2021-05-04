import sys
import time
from io import StringIO
import numpy as np

import tens_fcts as tf
import MPX
import PEPX
import PEPS_env as ENV
import PEP0_env as ENV0



#### MPO block functions:  add, subtract, outer product, scaling, compression ####
#### all matrices are [left,right,down,up] -> Dw x Dw x d x d                 ####

paulis = {}
paulis['ID'] = np.array([[1.,0],[0,1.]])
paulis['SX'] = np.array([[0,1.],[1.,0]])   *1./2.
paulis['SY'] = np.array([[0,-1.j],[1.j,0]])*1./2.
paulis['SZ'] = np.array([[1.,0],[0,-1.]])  *1./2.

sx = np.array([[0.,1.] ,[1.,0.]]) *1./2.
sy = np.array([[0,-1.j],[1.j,0]]) *1./2.
sz = np.array([[1.,0.] ,[0.,-1.]])*1./2.
s2 = np.array([[1.,0.] ,[0.,1.]])
sp = np.array([[0.,1.] ,[0.,0.]]) *1./np.sqrt(2.)
sm = np.array([[0.,0.] ,[1.,0.]]) *1./np.sqrt(2.)


####  defintion as sum of trotter steps ####
class TrotterOp:
    def __init__(self,Ls,db=2):

        # trotter steps are lists of 1-D MPOs

        # dictionary keys are the connectivitiy (eg. 'lir', 'r',...)
        # note that starts at inner left corner (0,0) ind
        self.Ls = Ls
        self.db = db
        self.it_sh = (1,1)  # block size for iterating trotter steps over lattice

        # mapping of trotter step anchored at lattice site (i,j)
        self.map = np.empty(Ls,dtype=list)
        for idx in np.ndindex(self.Ls):
            self.map[idx] = ['B']

        ## dictionarey of key (eg 'B' for bulk) to MPO, connectivity, shape of operator
        self.ops  = { 'B': MPX.MPX( [np.eye(db).reshape(1,db,db,1)] )  }
        self.conn = { 'B': '' }
        self.ns   = { 'B': (1,1) }
        self.ind0 = { 'B': (0,0) }


    def get_trotter_list(self):
        ''' returns list with trotter steps [ ((lattice site), operator) ]
        '''
        op_list = []
        for idx in np.ndindex(self.Ls):
            op_list += [ (idx, m_op) for m_op in self.map[idx] ]
        return op_list


    # def meas_H(self,pepx,pepx_type='peps',bounds=None,XMAX=100):
    #     ''' bounds:  [subenvLs, envI, envO, subenvRs]  '''
    #     if pepx_type in ['peps','state']:
    #         return self.meas_H_peps(pepx,bounds,XMAX)
    #     elif pepx_type in ['DM','dm','pepo','rho']:
    #         return self.meas_H_rho(pepx,bounds,XMAX)

    # # @profile
    # def meas_H_peps(self,pepx,bounds=None,XMAX=100):
    #     # pepx could also be pepo ('dm','DM','rho','pepo')

    #     L1, L2 = self.Ls
    #     NR, NC = self.it_sh

    #     # calculate envs and sweep through
    #     if bounds is None:

    #         # print 'measH', [m.shape for idx,m in np.ndenumerate(pepx)]

    #         envIs = ENV.get_boundaries( np.conj(pepx), pepx, 'I', L1, XMAX=XMAX)  # list of len ii+1
    #         envOs = ENV.get_boundaries( np.conj(pepx), pepx, 'O', 0 , XMAX=XMAX)  # list of len L+1

    #         senvRs = []
    #         senvLs = []
    #         for i in range(L1):
    #             NR_ = min(NR,L1-i)
    #             senvRs.append(ENV.get_subboundaries(envIs[i],envOs[i+NR_],
    #                                                 np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'R',0 ,XMAX=XMAX))
    #             senvLs.append(ENV.get_subboundaries(envIs[i],envOs[i+NR_],
    #                                                 np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'L',L2,XMAX=XMAX))
    #     else:
    #         senvLs, envIs, envOs, senvRs = bounds

    #         if senvLs is None:
    #             for i in range(L1):
    #                 NR_ = min(NR,L1-i)
    #                 senvRs.append(ENV.get_subboundaries(envIs[i],envOs[i+NR_],
    #                                                     np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'R',0 ,XMAX=XMAX))

    #         if senvRs is None:
    #             for i in range(L1):
    #                 NR_ = min(NR,L1-i)
    #                 senvLs.append(ENV.get_subboundaries(envIs[i],envOs[i+NR_],
    #                                                 np.conj(pepx[i:i+NR_,:]),pepx[i:i+NR_,:],'L',L2,XMAX=XMAX))
 

    #      
    #     op_inds = {}            # elements to call for each distinct op_conn
    #     for opk in self.ops.keys():
    #         try:
    #             xx = op_inds[self.conn[opk]]
    #         except(KeyError):
    #             inds_list = PEPX.get_conn_inds(self.conn[opk],self.ind0[opk])
    #             op_inds[self.conn[opk]] = inds_list
    #             # print self.conn[opk], inds_list


    #     obs_val = 0.
    #     for idx, m_op in self.get_trotter_list():

    #         i,j = idx
    #         NR_ = min(NR,L1-i)
    #         NC_ = min(NC,L2-j)

    #         sub_pepx = pepx[i:i+NR_,j:j+NC_]

    #         bi = envIs[i][j:j+NC_]
    #         bo = envOs[i+NR_][j:j+NC_]

    #         bl = senvLs[i][j]
    #         br = senvRs[i][j+NC_] 

    #         # xs = (0,)
    #         # ys = (0,)
    #         # opi = op_inds[self.conn[m_op]]
    #         # for ind in opi:
    #         #     xs = xs + ( xs[-1]+ind[0], )
    #         #     ys = ys + ( ys[-1]+ind[1], )
    #         xs, ys = op_inds[self.conn[m_op]]

    #         # print 'meas H', idx, m_op,xs,ys

    #         # pepx_list, axTs = pepx.get_sites(sub_pepx,self.ind0[m_op],self.conn[m_op])
    #         pepx_list, axTs = PEPX.connect_pepx_list(sub_pepx[xs,ys], self.conn[m_op])
    #         app_list, errs = PEPX.mpo_update(pepx_list,None,self.ops[m_op],DMAX=XMAX)

    #         app_pepx = sub_pepx.copy()
    #         app_pepx[xs,ys] = PEPX.transpose_pepx_list(app_list, axTs)
    #         
    #         exp_val = ENV.embed_sites_ovlp(np.conj(sub_pepx),app_pepx,[bl,bi,bo,br],XMAX=XMAX)
    #         
    #         obs_val += exp_val

    #     return obs_val

    # # @profile
    # def meas_H_rho(self,pepx,bounds=None,XMAX=100):
    #     # pepx could also be pepo ('dm','DM','rho','pepo')

    #     L1, L2 = self.Ls
    #     NR, NC = self.it_sh

    #     trPEPO = ENV0.trace(pepx)

    #     # calculate envs and sweep through
    #     if bounds is None:
    #         envIs = ENV0.get_boundaries( trPEPO, 'I', L1, XMAX=XMAX)  # list of len ii+1
    #         envOs = ENV0.get_boundaries( trPEPO, 'O', 0 , XMAX=XMAX)  # list of len L+1

    #         senvRs = []
    #         senvLs = []
    #         for i in range(L1):
    #             NR_ = min(NR,L1-i)
    #             senvRs.append(ENV0.get_subboundaries(envIs[i],envOs[i+NR_],trPEPO[i:i+NR_,:],'R',0 ,XMAX=XMAX))
    #             senvLs.append(ENV0.get_subboundaries(envIs[i],envOs[i+NR_],trPEPO[i:i+NR_,:],'L',L2,XMAX=XMAX))
    #     else:
    #         senvLs, envIs, envOs, senvRs = bounds

    #      
    #     op_inds = {}            # elements to call for each distinct op_conn
    #     for opk in self.ops.keys():
    #         try:
    #             xx = op_inds[self.conn[opk]]
    #         except(KeyError):
    #             inds_list = PEPX.get_conn_inds(self.conn[opk],self.ind0[opk])
    #             op_inds[self.conn[opk]] = inds_list
    #             # print self.conn[opk], inds_list

    #     obs_val = 0.
    #     for idx, m_op in self.get_trotter_list():
    #         i,j = idx
    #         NR_ = min(NR,L1-i)
    #         NC_ = min(NC,L2-j)

    #         sub_pepx = pepx[i:i+NR_,j:j+NC_]

    #         bi = envIs[i][j:j+NC_]
    #         bo = envOs[i+NR_][j:j+NC_]

    #         bl = senvLs[i][j]
    #         br = senvRs[i][j+NC_] 

    #         xs, ys = op_inds[self.conn[m_op]]
    #         pepx_list, axTs = PEPX.connect_pepx_list(sub_pepx[xs,ys], self.conn[m_op])
    #         app_list, errs = PEPX.mpo_update(pepx_list,None,self.ops[m_op],DMAX=XMAX)

    #         app_pepx = sub_pepx.copy()
    #         app_pepx[xs,ys] = PEPX.transpose_pepx_list(app_list, axTs)
    #         
    #         exp_val  = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='I',XMAX=XMAX)
    #         # exp_val1 = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='O',XMAX=XMAX)
    #         # exp_val2 = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='L',XMAX=XMAX)
    #         # exp_val3 = ENV0.embed_sites_contract(ENV0.trace(app_pepx),[bl,bi,bo,br],side='R',XMAX=XMAX)

    #         # diffs = np.abs([exp_val-exp_val1, exp_val-exp_val2, exp_val-exp_val3, exp_val1-exp_val2, exp_val1-exp_val3, exp_val2-exp_val3])
    #         # print diffs
    #         # if np.any(diffs > 1.0e-8):
    #         #    print 'embed sites error'
    #         #    exit()

    #         obs_val += exp_val

    #     return obs_val



def mpo_to_trotterH(Ls,H_MPO,db=2,ind0=(0,0)):

    L1,L2 = Ls
    L = len(H_MPO)

    assert(L1==1 or L2==1),'should be 1D peps'

    trotterH = TrotterOp(Ls,db)

    trotterH.Ls = Ls
    trotterH.db = db
    trotterH.it_sh = Ls

    trotterH.map = np.empty(Ls,dtype=list)
    for idx in np.ndindex(Ls):    trotterH.map[idx] = []
    trotterH.map[0,0] = ['op']

    trotterH.ops  = {'op': H_MPO }
    trotterH.ind0 = {'op': ind0}

    if L1==1:
        trotterH.conn = {'op': 'r'*(L-1)}
        trotterH.ns   = {'op': (1,L)}
    elif L2==1:
        trotterH.conn = {'op': 'o'*(L-1)}
        trotterH.ns   = {'op': (L,1)}

    return trotterH


       
class Heisenberg_sum(TrotterOp):      # even on-site weighting
    '''
    sum_i h_i S_i + J_i (S_i S_(i+1))
    '''
    def __init__(self,Ls,hs=None,Js=None):
        
        L1, L2 = Ls

        self.Ls = Ls
        self.db = 2
        self.it_sh = (min(2,L1),min(2,L2))

        if hs == None and Js == None:   # default AFM heisenberg 
            op_1 = np.zeros((1,2,2,3))
            op_2 = np.zeros((3,2,2,1))
            op_1[0,:,:,0] = sp.copy()
            op_1[0,:,:,1] = sm.copy()
            op_1[0,:,:,2] = sz.copy()
            op_2[0,:,:,0] = sm.copy()
            op_2[1,:,:,0] = sp.copy()
            op_2[2,:,:,0] = sz.copy()

            trotter_op = MPX.MPX( [op_1, op_2],  phys_bonds=[(2,2),(2,2)] )
            self.ops  = {'hB': trotter_op.copy(), 'vB': trotter_op.copy()}
            self.conn = {'hB': 'r', 'vB': 'o'}
            self.ns   = {'hB': (1,2), 'vB': (2,1)}

        else:
            ops = paulis
            ops['SP'] = sp
            ops['SM'] = sm

            # hs = x, y, z
            hs = np.array(hs)
            singList = ['SX','SY','SZ']
            # Js = xx, yy, zz, xy, yz, zx
            Js = np.array(Js)
            if Js[0] == Js[1] and len(Js) <= 3:
                print 'X, Y -> +, -'
                coupList = [('SP','SM'),('SM','SP'),('SZ','SZ')]
                dtype = np.float64
            else:
                coupList = [('SX','SX'),('SY','SY'),('SZ','SZ'),('SX','SY'),('SY','SZ'),('SZ','SX')]
                dtype = np.complex128
            
            if len(Js) <= 3:
                Js1 = np.where(Js != 0)[0]
                Js2 = []
                Dw = len(Js1) + 2    # put on-site term only on op1, not op2
            else:
                Js1 = np.where(np.array(Js[:3]) != 0)[0]
                Js2 = np.where(np.array(Js[3:]) != 0)[0]
                Dw = len(Js1) + len(Js2)*2 + 2

            op_1  = np.zeros((1,2,2,Dw),dtype=dtype)
            op_2  = np.zeros((Dw,2,2,1),dtype=dtype)
            
            # identity terms
            op_1[0,:,:,-1] = np.eye(2)
            op_2[0,:,:,0]  = np.eye(2) 
            

            for ind in range(3):         # X, Y, Z  on-site terms
                if not hs[ind] == 0:
                    op_1[0,:,:,0] = op_1[0,:,:,0].copy()  + 0.25*hs[ind]*ops[singList[ind]].copy()
            op_2[-1,:,:,0] = op_1[0,:,:,0].copy()


            for ii in range(len(Js1)):   # XX, YY, ZZ couplings
                ind = Js1[ii]
                op_1[0,:,:,1+ii] = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+ii,:,:,0] = ops[coupList[ind][0]].copy()
                
            for ii in range(len(Js2)):   # XY, YZ, ZX couplings
                ind = Js2[ii]+3
                op_2[1+len(Js1)+2*ii,:,:,0]   = ops[coupList[ind][0]].copy()
                op_1[0,:,:,1+len(Js1)+2*ii]   = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+len(Js1)+2*ii+1,:,:,0] = ops[coupList[ind][1]].copy()
                op_1[-1,:,:,1+len(Js1)+2*ii]  = ops[coupList[ind][0]].copy()*Js[ind]

            # print 'heisenberg even'
            # for j in np.ndindex(Dw):
            #     print op_1[0,:,:,j]
            # for j in np.ndindex(Dw):
            #     print op_2[j,:,:,0]

            # make exception trotter operators
            ## bulk trotter mpos:   on site weights:  (1/4, 1/4) for 'r' and 'o'
            ## edges:               on site weights:  (3/8, 3/8) for 'o' or 'r'
            ## corners:             on site weights:  (1/2, 3/8) for 'r' or 'o'

            # define self.map:  2darray, el i,j = list of trotter steps starting at site (i,j)
            # define self.ops:  dictionary of trotter step operators
            self.map = np.empty(Ls,dtype=list)
            if L1 == 1 and L2 == 1:
                self.map[0,0] = ['o']
                self.ops  = {'o': MPX.MPX( op_1[0,:,:,0] * 2 ) }
                self.conn = {'o': ''}
                self.ns   = {'o': (1,1)}
                self.ind0 = {'o': (0,0)}

            elif L1 == 1 or L2==1:     # row or col
                op_10 = op_1.copy()
                op_10[0,:,:,0]  *= 4

                op_2L = op_2.copy()
                op_2L[-1,:,:,0] *= 4

                op_1[ 0,:,:,0] *= 2
                op_2[-1,:,:,0] *= 2


                for idx in np.ndindex(Ls):   self.map[idx] = ['tB']
                self.ops  = {'t0': MPX.MPX( [op_10,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                             'tB': MPX.MPX( [op_1 ,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                             'tL': MPX.MPX( [op_1 ,op_2L], phys_bonds=[(2,2),(2,2)] )}

                if L1 == 1:   # row
                    self.map[0, 0] = ['t0']
                    self.map[0,-2] = ['tL']
                    self.map[0,-1] = []
                    self.conn = {'t0': 'r', 'tB': 'r', 'tL': 'r'}
                    self.ns   = {'t0': (1,2), 'tB': (1,2), 'tL': (1,2)}
                    self.ind0 = {'t0': (0,0), 'tB': (0,0), 'tL': (0,0)}
                else:
                    self.map[ 0,0] = ['t0']
                    self.map[-2,0] = ['tL']
                    self.map[-1,0] = []
                    self.conn = {'t0': 'o', 'tB': 'o', 'tL': 'o'}
                    self.ns   = {'t0': (2,1), 'tB': (2,1), 'tL': (2,1)}
                    self.ind0 = {'t0': (0,0), 'tB': (0,0), 'tL': (0,0)}

            elif L1 == 2 and L2 == 2:      # 2x2 lattice
                op_1C = op_1.copy()
                op_1C[0,:,:,0]  *= 2

                op_2C = op_2.copy()
                op_2C[-1,:,:,0] *= 2

                for idx in np.ndindex(Ls):   self.map[idx] = ['hC','vC']
                self.map[0,0] = ['hC','vC']
                self.map[0,1] = ['vC']
                self.map[1,0] = ['hC']
                self.map[1,1] = []
                self.ops  = {'hC': MPX.MPX( [op_1C,op_2C], phys_bonds=[(2,2),(2,2)] ),
                             'vC': MPX.MPX( [op_1C,op_2C], phys_bonds=[(2,2),(2,2)] )}
                self.conn = {'hC': 'r', 'vC': 'o'}
                self.ns   = {'hC': (1,2), 'vC': (2,1)}
                self.ind0 = {'hC': (0,0), 'vC': (0,0)}

            else:
                op_1E = op_1.copy()
                op_1E[0,:,:,0]  = 1.5*op_1[0,:,:,0].copy()
                op_2E = op_2.copy()
                op_2E[-1,:,:,0] = 1.5*op_2[-1,:,:,0].copy()

                op_1C = op_1.copy()   
                op_1C[0,:,:,0]  = 2.*op_1[0,:,:,0].copy()  
                op_2C = op_2.copy()    
                op_2C[-1,:,:,0] = 2.*op_2[-1,:,:,0].copy() 

                for idx in np.ndindex(Ls): 
                    # catch all corners first
                    if   idx == (0,0):        self.map[idx] = ['hC1','vC1'] 
                    elif idx == (0,L2-1):     self.map[idx] = ['vC1']         # top right corner
                    elif idx == (0,L2-2):     self.map[idx] = ['hC2','vB']
                    elif idx == (L1-2,0):     self.map[idx] = ['hB' ,'vC2']   # bottom left corner
                    elif idx == (L1-1,0):     self.map[idx] = ['hC1']
                    elif idx == (L1-2,L2-1):  self.map[idx] = ['vC2']         # bottom right corner
                    elif idx == (L1-1,L2-1):  self.map[idx] = []
                    elif idx == (L1-1,L2-2):  self.map[idx] = ['hC2']
                    # then catch rows/cols
                    elif idx[1] == 0:         self.map[idx] = ['vE','hB']     # left col
                    elif idx[0] == 0:         self.map[idx] = ['hE','vB']     # top row
                    elif idx[1] == L2-1:      self.map[idx] = ['vE']          # right col
                    elif idx[0] == L1-1:      self.map[idx] = ['hE']          # bottom row
                    # then everything else
                    else:                    self.map[idx] = ['hB','vB']     # interior bonds
     
                self.ops = {'hB' :  MPX.MPX( [op_1 ,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                            'vB' :  MPX.MPX( [op_1 ,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                            'hE' :  MPX.MPX( [op_1E,op_2E], phys_bonds=[(2,2),(2,2)] ),
                            'vE' :  MPX.MPX( [op_1E,op_2E], phys_bonds=[(2,2),(2,2)] ),
                            'hC1':  MPX.MPX( [op_1C,op_2E], phys_bonds=[(2,2),(2,2)] ),
                            'vC1':  MPX.MPX( [op_1C,op_2E], phys_bonds=[(2,2),(2,2)] ),
                            'hC2':  MPX.MPX( [op_1E,op_2C], phys_bonds=[(2,2),(2,2)] ),
                            'vC2':  MPX.MPX( [op_1E,op_2C], phys_bonds=[(2,2),(2,2)] )}

                if L1 == 2:  self.ops['vC1'] = MPX.MPX( [op_1C,op_2C], phys_bonds=[(2,2),(2,2)] )
                if L2 == 2:  self.ops['hC1'] = MPX.MPX( [op_1C,op_2C], phys_bonds=[(2,2),(2,2)] )

                self.conn = {'hB':'r','vB':'o','hE':'r','vE':'o','hC1':'r','hC2':'r','vC1':'o','vC2':'o'}
                self.ns   = {'hB':(1,2),'vB':(2,1),'hE':(1,2),'vE':(2,1),'hC1':(1,2),'vC1':(2,1),'hC2':(1,2),'vC2':(2,1)}
                self.ind0 = {'hB':(0,0),'vB':(0,0),'hE':(0,0),'vE':(0,0),'hC1':(0,0),'vC1':(0,0),'hC2':(0,0),'vC2':(0,0)}

        print 'Heisenberg map',  Ls, self.map    
        # print exit()


class Heisenberg_sum_uneven(TrotterOp):   
    '''
    sum_i h_i S_i + J_i (S_i S_(i+1))
    '''
    def __init__(self,Ls,hs=None,Js=None):
        
        L1, L2 = Ls

        self.Ls = Ls
        self.db = 2
        self.it_sh = (min(2,L1),min(2,L2))

        if hs == None and Js == None:   # default AFM heisenberg 
            op_1 = np.zeros((1,2,2,3))
            op_2 = np.zeros((3,2,2,1))
            op_1[0,:,:,0] = sp.copy()
            op_1[0,:,:,1] = sm.copy()
            op_1[0,:,:,2] = sz.copy()
            op_2[0,:,:,0] = sm.copy()
            op_2[1,:,:,0] = sp.copy()
            op_2[2,:,:,0] = sz.copy()

            trotter_op = MPX.MPX( [op_1, op_2],  phys_bonds=[(2,2),(2,2)] )
            self.ops  = {'hB': trotter_op.copy(), 'vB': trotter_op.copy()}
            self.conn = {'hB': 'r', 'vB': 'o'}
            self.ns   = {'hB': (1,2), 'vB': (2,1)}
            self.ind0 = {'hB': (0,0), 'vB': (0,0)}

        else:
            ops = paulis
            ops['SP'] = sp
            ops['SM'] = sm

            # hs = x, y, z
            hs = np.array(hs)
            singList = ['SX','SY','SZ']
            # Js = xx, yy, zz, xy, yz, zx
            Js = np.array(Js)
            coupList = [('SX','SX'),('SY','SY'),('SZ','SZ'),('SX','SY'),('SY','SZ'),('SZ','SX')]
            # coupList = [('SP','SM'),('SM','SP'),('SZ','SZ'),('SX','SY'),('SY','SZ'),('SZ','SX')]
            
            if len(Js) <= 3:
                Js1 = np.where(Js != 0)[0]
                Js2 = []
                Dw = len(Js1) + 1    # put on-site term only on op1, not op2
            else:
                Js1 = np.where(np.array(Js[:3]) != 0)[0]
                Js2 = np.where(np.array(Js[3:]) != 0)[0]
                Dw = len(Js1) + len(Js2)*2 + 1

            op_1  = np.zeros((1,2,2,Dw),dtype=np.complex128)
            op_2  = np.zeros((Dw,2,2,1),dtype=np.complex128)
            
            # identity terms
            op_2[0,:,:,0]  = np.eye(2) 
            

            for ind in range(3):         # X, Y, Z  on-site terms
                if not hs[ind] == 0:
                    op_1[0,:,:,0] = op_1[0,:,:,0].copy()  + 0.5*hs[ind]*ops[singList[ind]].copy()

            for ii in range(len(Js1)):   # XX, YY, ZZ couplings
                ind = Js1[ii]
                print 'Js1',ind
                op_1[0,:,:,1+ii] = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+ii,:,:,0] = ops[coupList[ind][0]].copy()
                
            for ii in range(len(Js2)):   # XY, YZ, ZX couplings
                ind = Js2[ii]+3
                op_2[1+len(Js1)+2*ii,:,:,0]   = ops[coupList[ind][0]].copy()
                op_1[0,:,:,1+len(Js1)+2*ii]   = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+len(Js1)+2*ii+1,:,:,0] = ops[coupList[ind][1]].copy()
                op_1[-1,:,:,1+len(Js1)+2*ii]  = ops[coupList[ind][0]].copy()*Js[ind]


            # make exception trotter operators
            ## bulk trotter mpos:   on site weights:  (1/2, 0) for 'r' and 'o'
            ## right col:           on site weights:  (1, 0)   for 'o'
            ## bottom col:          on site weights:  (1, 0)   for 'r'
            ## bottom-right corner: on site weights:  (1, 1)   for 'r'  
            op_1L = op_1.copy()
            op_1L[0,:,:,0] = 2*op_1[0,:,:,0].copy()

            op_1C = np.zeros((1,2,2,Dw+1),dtype=np.complex128)    # bottom right corner op (1,1)
            op_2C = np.zeros((Dw+1,2,2,1),dtype=np.complex128)
            op_1C[:,:,:,:-1] = op_1L.copy()   # on site term has coeff 1 (so use op_1L instead of op_1)
            op_1C[0,:,:,-1]  = np.eye(2)      # need identity for on-site term in op_2C
            op_2C[:-1,:,:,:] = op_2.copy()    # only ok bc op_2C is Dw x d x d x 1
            op_2C[-1,:,:,0]  = op_1L[0,:,:,0].copy()   # on site term


            # print 'heisenberg'
            # for j in np.ndindex(Dw):
            #     print op_1[0,:,:,j]
            # for j in np.ndindex(Dw):
            #     print op_2[j,:,:,0]
            # print 'heisenberg'
            # for j in np.ndindex(Dw+1):
            #     print op_1C[0,:,:,j]
            # for j in np.ndindex(Dw+1):
            #     print op_2C[j,:,:,0]

            # define self.map:  2darray, el i,j = list of trotter steps starting at site (i,j)
            # define self.ops:  dictionary of trotter step operators
            self.map = np.empty(Ls,dtype=list)
            if L1 == 1 and L2 == 1:
                self.map[0,0] = ['o']
                self.ops  = {'o': MPX.MPX( op_1[0,:,:,0] * 2 ) }
                self.conn = {'o': ''}
                self.ns   = {'o': (1,1)}
                self.ind0 = {'o': (0,0)}
            elif L1 == 1:     # row 
                for idx in np.ndindex(Ls):   self.map[idx] = ['hB']
                self.map[0,-2] = ['hL']
                self.map[0,-1] = []
                self.ops  = {'hB': MPX.MPX( [op_1L,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                             'hL': MPX.MPX( [op_1C,op_2C], phys_bonds=[(2,2),(2,2)] )}
                self.conn = {'hB': 'r', 'hL': 'r'}
                self.ns   = {'hB': (1,2), 'hL': (1,2)}
                self.ind0 = {'hB': (0,0), 'hL': (0,0)}
            elif L2 == 1:     # col
                for idx in np.ndindex(Ls):   self.map[idx] = ['vB']
                self.map[-2][0] = ['vL']
                self.map[-1][0] = []
                self.ops  = {'vB': MPX.MPX( [op_1L,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                             'vL': MPX.MPX( [op_1C,op_2C], phys_bonds=[(2,2),(2,2)] )}
                self.conn = {'vB': 'o','vL': 'o'}
                self.ns   = {'vB': (2,1), 'vL': (2,1)}
                self.ind0 = {'vB': (0,0), 'vL': (0,0)}
            else:
                for idx in np.ndindex(Ls):     self.map[idx] = ['hB','vB']
                for i in range(L1):            self.map[i,-1] = ['vL']   # right col
                for j in range(L2):            self.map[-1,j] = ['hL']   # bottom row
                self.map[-1,-2] = ['hLL']   # bottom right corner (L,L-1) -> (L,L) 
                self.map[-1,-1] = []        # no trotter steps anchored on (L,L) lattice site
               
                   
                self.ops = {'hB':  MPX.MPX( [op_1 ,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                            'vB':  MPX.MPX( [op_1 ,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                            'hL':  MPX.MPX( [op_1L,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                            'vL':  MPX.MPX( [op_1L,op_2 ], phys_bonds=[(2,2),(2,2)] ),
                            'hLL': MPX.MPX( [op_1C,op_2C], phys_bonds=[(2,2),(2,2)] )}

                self.conn = {'hB':'r', 'vB':'o', 'hL':'r', 'vL': 'o', 'hLL': 'r' }
                self.ns   = {'hB':(1,2), 'vB':(2,1), 'hL':(1,2), 'vL':(2,1), 'hLL':(1,2)}
                self.ind0 = {'hB':(0,0), 'vB':(0,0), 'hL':(0,0), 'vL':(0,0), 'hLL':(0,0)}

            print self.map


class NNN_swap_row(TrotterOp):

    def __init__(self,Ls,set_num):
        
        L1, L2 = Ls

        self.Ls = Ls
        self.db = 2
        self.it_sh = (2,2)

        db = self.db
        swap_op  = np.einsum('ab,ij->aijb',np.eye(db),np.eye(db)).reshape(1,db,db,db,db,1)
        swap_MPO = MPX.MPX( tf.decompose_block(swap_op,2,0,-1,svd_str='ijk,...'), phys_bonds=[(db,db),(db,db)] )

        self.ops  = {'swap': swap_MPO}
        self.conn = {'swap':'o'}
        self.ns   = {'swap':(2,1)}
        self.ind0 = {'swap':(0,0)}

        self.map = np.empty(Ls,dtype=list)
        for idx in np.ndindex(Ls):   self.map[idx] = []
        if set_num == 0:
            for i in range(0,L1-1,2):
                for j in range(1,L2,2):
                    self.map[i,j] = ['swap']
        elif set_num == 1:
            for i in range(1,L1-1,2):
                for j in range(1,L2,2):
                    self.map[i,j] = ['swap']


class NNN_t2_row(TrotterOp):

    def __init__(self,Ls,Js,set_num):

        L1, L2 = Ls

        self.Ls = Ls
        self.db = 2
        self.it_sh = (2,2)

        ops = paulis
        ops['SP'] = sp
        ops['SM'] = sm


        if L1 == 1 or L2==1:    # return heisenberg
            heis_pepo = {} 
            self.ops  = {} 
            self.ns   = {} 
            self.conn = {} 
            self.map  = np.empty(Ls,dtype=list)
            self.it_sh = {}

        else: 

            ## no on-site coupling
            # Js = xx, yy, zz, xy, yz, zx
            Js = np.array(Js)
            if Js[0] == Js[1] and len(Js) <= 3:
                print 'X, Y -> +, -'
                coupList = [('SP','SM'),('SM','SP'),('SZ','SZ')]
                dtype = np.float64
            else:
                coupList = [('SX','SX'),('SY','SY'),('SZ','SZ'),('SX','SY'),('SY','SZ'),('SZ','SX')]
                dtype = np.complex128
            
            if len(Js) <= 3:
                Js1 = np.where(Js != 0)[0]
                Js2 = []
                Dw = len(Js1) + 2    # put on-site term only on op1, not op2
            else:
                Js1 = np.where(np.array(Js[:3]) != 0)[0]
                Js2 = np.where(np.array(Js[3:]) != 0)[0]
                Dw = len(Js1) + len(Js2)*2 + 2

            op_1  = np.zeros((1,2,2,Dw),dtype=dtype)
            op_2  = np.zeros((Dw,2,2,1),dtype=dtype)
            
            # identity terms
            op_1[0,:,:,-1] = np.eye(2)
            op_2[0,:,:,0]  = np.eye(2) 
            
            for ii in range(len(Js1)):   # XX, YY, ZZ couplings
                ind = Js1[ii]
                op_1[0,:,:,1+ii] = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+ii,:,:,0] = ops[coupList[ind][0]].copy()
                
            for ii in range(len(Js2)):   # XY, YZ, ZX couplings
                ind = Js2[ii]+3
                op_2[1+len(Js1)+2*ii,:,:,0]   = ops[coupList[ind][0]].copy()
                op_1[0,:,:,1+len(Js1)+2*ii]   = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+len(Js1)+2*ii+1,:,:,0] = ops[coupList[ind][1]].copy()
                op_1[-1,:,:,1+len(Js1)+2*ii]  = ops[coupList[ind][0]].copy()*Js[ind]

            ## no exception trotter operators bc no on-site field
            self.ops  = {'t2': MPX.MPX( [op_1,op_2] ) }
            self.conn = {'t2': 'r'}
            self.ns   = {'t2': (1,2)}
            self.ind0 = {'t2': (0,0)}

            self.map = np.empty(Ls,dtype=list)
            for idx in np.ndindex(Ls):   self.map[idx] = []
            if set_num == 0:
                for i in range(0,L1-L1%2):      # if even, include last row; if odd don't include last row
                    for j in range(0,L2-1):
                        self.map[i,j] = ['t2']
            elif set_num == 1:
                for i in range(1,L1-(L1+1)%2):  # if even, don't include last row; if odd include last row
                    for j in range(0,L2-1):
                        self.map[i,j] = ['t2']



class t1t2_3body_sum(TrotterOp):
    '''
    sum_<<ij>> Xi Xj + Yi Yj  + gamma*sum_i Xi + tau*sum_i Zi

    as an 4-term MPO x -- x
                          |
                     x -- x

    '''
    def __init__(self,Ls,hs,t1s,t2s):
        
        L1, L2 = Ls

        self.Ls = Ls
        self.db = 2
        self.it_sh = (2,2)

        ops = paulis
        ops['SP'] = sp
        ops['SM'] = sm


        if L1 == 1 or L2==1:    # return heisenberg
            heis_pepo = Heisenberg_sum(Ls,hs,t1s)
            self.ops  = heis_pepo.ops
            self.ns   = heis_pepo.ns
            self.conn = heis_pepo.conn
            self.map  = heis_pepo.map
            self.it_sh = heis_pepo.it_sh

        else: 

            ### "bulk" mpos:  4 MPOs ('ro','or','lo','ri')
            ### on site:  1/16, 1/8, 1/16
            ### coupling:  1/4, 1/4
            ### nnn:  1
    
            #### getting on site  hs = x, y, z  #########
            opList = ['SX','SY','SZ']

            onsite = np.zeros((2,2))
            for ind in range(3):         # X, Y, Z  on-site terms
                print ind, hs[ind]
                onsite = onsite + hs[ind]*ops[opList[ind]].copy()

            if hs[1] != 0:
                dtype_h = np.complex128
            else:
                dtype_h = np.float64
                onsite = np.real(onsite)
             

            ####  getting couplings #######
            if t1s[0] == t1s[1] and t2s[0] == t2s[1]:
                print 'X, Y -> +, -'
                coupList = [('SP','SM'),('SM','SP'),('SZ','SZ')]
                dtype_t = np.float64
            else:
                coupList = [('SX','SX'),('SY','SY'),('SZ','SZ')]
                dtype_t = np.complex128

            dtype = np.result_type(dtype_h,dtype_t)
    
            
            t_inds = np.where(np.abs(np.array(t1s))+np.abs(np.array(t2s)) != 0)[0]   # t1s or t2s != 0
            n1 = len(t_inds) 
            Dw = n1 + 2
            # even when t2=0, need extra space for the long range interaction from 1--4

            # build the actual matrices
            op1 = np.zeros((1 ,2,2,Dw), dtype=dtype)
            opm = np.zeros((Dw,2,2,Dw), dtype=dtype)
            opL = np.zeros((Dw,2,2,1 ), dtype=dtype)
    
    
            # on site terms
            op1[ 0,:,:,0] = onsite.copy()
            opm[-1,:,:,0] = onsite.copy()
            opL[-1,:,:,0] = onsite.copy()

            # nn identities
            op1[ 0,:,:,-1] = np.eye(2) 
            opm[ 0,:,:, 0] = np.eye(2) 
            opm[-1,:,:,-1] = np.eye(2) 
            opL[ 0,:,:, 0] = np.eye(2) 
    
            i1 = 0
            for ti in t_inds:   # t1 couplings

                x1 = 1+i1

                print coupList[ti]

                # operator 1
                op1[ 0,:,:,x1] = ops[coupList[ti][0]]                 # operators on bottom row

                # middle operators
                opm[x1,:,:, 0] = ops[coupList[ti][1]]*t1s[ti]         # operators on left col
                opm[x1,:,:,x1] = np.eye(2)*t2s[ti]                    # long range identities
                opm[-1,:,:,x1] = ops[coupList[ti][0]]*t1s[ti]         # bottom row

                # operator 4 (last)
                opL[x1,:,:, 0] = ops[coupList[ti][1]]

                i1 += 1      


            basic_op = MPX.MPX( [op1,opm.copy(),opL], phys_bonds=[(2,2)]*3 )
            
            def mpo_scale_consts(scale_onsite=[1.,1.,1.],scale_nn=[1.,1.],scale_nnn=[1.]):
                # lists of length 3,2,1
                mpo_new = MPX.copy(basic_op)
                
                # on-site term scaling
                mpo_new[0][-1,:,:,0] *= scale_onsite[0]
                mpo_new[1][-1,:,:,0] *= scale_onsite[1]
                mpo_new[2][-1,:,:,0] *= scale_onsite[2]

                # nn bond scaling
                mpo_new[1][1:n1+1,:,:,0]  *= scale_nn[0]
                mpo_new[1][-1,:,:,1:n1+1] *= scale_nn[1]

                # nnn bond scaling
                for xx in range(1,n1+1):
                    mpo_new[1][xx,:,:,xx] *= scale_nnn[0]

                return mpo_new

    
            bulk_op = mpo_scale_consts([1./16,1./8,1./16],[1./4,1./4],[1./2])


            # print Dw
            # for i in range(Dw):
            #     print op1[0,:,:,i]

            ### border/exception mpos:
            self.map = np.empty((L1,L2),dtype=list)
            for idx in np.ndindex((L1-1,L2-1)):   self.map[idx] = ['BRO','BOR','BLO','BOL']
            self.ops  = {'BRO': bulk_op, 'BOR': bulk_op, 'BLO': bulk_op, 'BOL': bulk_op}
            self.conn = {'BRO': 'ro',  'BOR': 'or',  'BLO': 'lo',  'BOL': 'ol'}
            self.ns   = {'BRO': (2,2), 'BOR': (2,2), 'BLO': (2,2), 'BOL': (2,2)}
            self.ind0 = {'BRO': (0,0), 'BOR': (0,0), 'BLO': (0,1), 'BOL': (0,1)}    # idx within self.ns for first point

            for i in range(L1):   self.map[i][-1] = []    # set right col, bottom row --> empty lists (since ns = 2,
            for j in range(L2):   self.map[-1][j] = []    # no trotter steps applied at these lattice sites

    
            if L1 > 2 and L2 > 2:   

                corner_Xxx = mpo_scale_consts([1./4,1./4,1./16],[1./2,1./4],[1./2])
                corner_xxX = mpo_scale_consts([1./16,1./4,1./4],[1./4,1./2],[1./2])
                corner_xXx = mpo_scale_consts([1./8,1./2,1./8], [1./2,1./2],[1./2])
                corner_xxx = mpo_scale_consts([1./8,1./8,1./8], [1./4,1./4],[1./2])

                self.map[0,0] = ['C0RO','C0OR','C0LO','C0OL']
                self.ops['C0RO'] = corner_Xxx
                self.ops['C0OR'] = corner_Xxx
                self.ops['C0LO'] = corner_xXx
                self.ops['C0OL'] = corner_xxx
                self.conn.update({'C0RO':'ro',  'C0OR':'or',  'C0LO': 'lo',  'C0OL': 'ol'})
                self.ind0.update({'C0RO':(0,0), 'C0OR':(0,0), 'C0LO': (0,1), 'C0OL': (0,1)})
                self.ns  .update({'C0RO':(2,2), 'C0OR':(2,2), 'C0LO': (2,2), 'C0OL': (2,2)})

                self.map[0,-2] = ['C1RO','C1OR','C1LO','C1OL']
                self.ops['C1RO'] = corner_xXx
                self.ops['C1OR'] = corner_xxx
                self.ops['C1LO'] = corner_Xxx
                self.ops['C1OL'] = corner_Xxx
                self.conn.update({'C1RO':'ro',  'C1OR':'or',  'C1LO': 'lo',  'C1OL': 'ol'})
                self.ind0.update({'C1RO':(0,0), 'C1OR':(0,0), 'C1LO': (0,1), 'C1OL': (0,1)})
                self.ns  .update({'C1RO':(2,2), 'C1OR':(2,2), 'C1LO': (2,2), 'C1OL': (2,2)})

                self.map[-2,0] = ['C2RO','C2OR','C2LO','C2OL']
                self.ops['C2RO'] = corner_xxx
                self.ops['C2OR'] = corner_xXx
                self.ops['C2LO'] = corner_xxX
                self.ops['C2OL'] = corner_xxX
                self.conn.update({'C2RO':'ro',  'C2OR':'or',  'C2LO': 'lo',  'C2OL': 'ol'})
                self.ind0.update({'C2RO':(0,0), 'C2OR':(0,0), 'C2LO': (0,1), 'C2OL': (0,1)})
                self.ns  .update({'C2RO':(2,2), 'C2OR':(2,2), 'C2LO': (2,2), 'C2OL': (2,2)})

                self.map[-2,-2] = ['C3RO','C3OR','C3LO','C3OL']
                self.ops['C3RO'] = corner_xxX
                self.ops['C3OR'] = corner_xxX
                self.ops['C3LO'] = corner_xxx
                self.ops['C3OL'] = corner_xXx
                self.conn.update({'C3RO':'ro',  'C3OR':'or',  'C3LO': 'lo',  'C3OL': 'ol'})
                self.ind0.update({'C3RO':(0,0), 'C3OR':(0,0), 'C3LO': (0,1), 'C3OL': (0,1)})
                self.ns  .update({'C3RO':(2,2), 'C3OR':(2,2), 'C3LO': (2,2), 'C3OL': (2,2)})
 

                ## edge boxes
                edge_bb = mpo_scale_consts([1./8, 1./8,1./16],[1./4, 1./4],[1./2])
                edge_Bb = mpo_scale_consts([1./8, 1./4,1./16],[1./2, 1./4],[1./2])

                for j in range(1,L2-2):
                    self.map[0,j] = ['EIRO','EIOR','EILO','EIOL']
                self.ops['EIRO'] = edge_Bb
                self.ops['EIOR'] = edge_bb
                self.ops['EILO'] = edge_Bb
                self.ops['EIOL'] = edge_bb
                self.conn.update({'EIRO':'ro', 'EIOR':'or', 'EILO':'lo', 'EIOL':'ol'})
                self.ind0.update({'EIRO':(0,0),'EIOR':(0,0),'EILO':(0,1),'EIOL':(0,1)})
                self.ns  .update({'EIRO':(2,2),'EIOR':(2,2),'EILO':(2,2),'EIOL':(2,2)})

                for i in range(1,L1-2):
                    self.map[i,0] = ['ELRO','ELOR','ELLO','ELOL']
                self.ops['ELRO'] = edge_bb
                self.ops['ELOR'] = edge_Bb
                self.ops['ELLO'] = edge_Bb
                self.ops['ELOL'] = edge_bb
                self.conn.update({'ELRO':'ro', 'ELOR':'or', 'ELLO':'ir', 'ELOL':'ri'})
                self.ind0.update({'ELRO':(0,0),'ELOR':(0,0),'ELLO':(1,0),'ELOL':(1,0)})
                self.ns  .update({'ELRO':(2,2),'ELOR':(2,2),'ELLO':(2,2),'ELOL':(2,2)})

                for j in range(1,L2-2):
                    self.map[-2,j] = ['EORO','EOOR','EOLO','EOOL']
                self.ops['EORO'] = edge_bb
                self.ops['EOOR'] = edge_Bb
                self.ops['EOLO'] = edge_bb
                self.ops['EOOL'] = edge_Bb
                self.conn.update({'EORO':'il', 'EOOR':'li', 'EOLO':'ir', 'EOOL':'ri'})
                self.ind0.update({'EORO':(1,1),'EOOR':(1,1),'EOLO':(1,0),'EOOL':(1,0)})
                self.ns  .update({'EORO':(2,2),'EOOR':(2,2),'EOLO':(2,2),'EOOL':(2,2)})

                for j in range(1,L1-2):
                    self.map[j,-2] = ['ERRO','EROR','ERLO','EROL']
                self.ops['ERRO'] = edge_Bb
                self.ops['EROR'] = edge_bb
                self.ops['ERLO'] = edge_bb
                self.ops['EROL'] = edge_Bb
                self.conn.update({'ERRO':'il', 'EROR':'li', 'ERLO':'lo', 'EROL':'ol'})
                self.ind0.update({'ERRO':(1,1),'EROR':(1,1),'ERLO':(0,1),'EROL':(0,1)})
                self.ns  .update({'ERRO':(2,2),'EROR':(2,2),'ERLO':(2,2),'EROL':(2,2)})

            
                # empty sites
                for i in range(L1):                    self.map[i,-1] = []
                for j in range(L2):                    self.map[-1,j] = []
 

            elif L1 == 2 and L2 == 2:    # bulk mpo --> boundary mpo
                
                # print 'basic 1', [m.transpose(0,3,1,2) for m in basic_op]
                new_op = mpo_scale_consts([1./4,1./2,1./4],[1./2,1./2],[1./2])

                # print 'basic 2', [m.transpose(0,3,1,2) for m in basic_op]
                # print 'new', [m.transpose(0,3,1,2) for m in new_op]

                self.map[0,0] = ['RO','OR','LO','OL']
                self.ops  = {'RO': new_op,'OR': new_op,'LO': new_op,'OL': new_op}
                self.conn = {'RO': 'ro',  'OR': 'or',  'LO': 'lo',  'OL': 'ol'}
                self.ind0 = {'RO': (0,0), 'OR': (0,0), 'LO': (0,1), 'OL': (0,1)}
                self.ns   = {'RO': (2,2), 'OR': (2,2), 'LO': (2,2), 'OL': (2,2)}

                self.map[0,1] = []
                self.map[1,1] = []
                self.map[1,0] = []

            elif L1 == 2:
 
                raise(NotImplementedError)
 
                sideL = MPX.copy(bulk_op)           # left side
                sideL[0][ 0,:,:,1:n1+1] *= 2        # couplings:  1,0.5,1,1
                sideL[2][-1,:,:,1:n1+1] *= 2
                sideL[0][-1,:,:,0] *= 4             # on-site:  1,0.5,0.5,1
                sideL[1][-1,:,:,0] *= 2
                sideL[2][-1,:,:,0] *= 2
                sideL[3][-1,:,:,0] *= 4
                
                sideR = MPX.copy(bulk_op)           # right side
                sideR[1][1:n1+1,:,:, 0] *= 2        # couplings:  1,1,1,0.5
                sideR[1][-1,:,:,1:n1+1] *= 2
                sideR[2][-1,:,:,1:n1+1] *= 2
                sideR[0][-1,:,:,0] *= 2             # on-site:  0.5,1,1,0.5
                sideR[1][-1,:,:,0] *= 4
                sideR[2][-1,:,:,0] *= 4
                sideR[3][-1,:,:,0] *= 2

                rows = MPX.copy(bulk_op)            # 2 rows
                rows[1][1:n1+1,:,:, 0] *= 2         # couplings:  1,0.5,1,0.5
                rows[2][-1,:,:,1:n1+1] *= 2
                for i in range(4):                  # on-site:    0.5,0.5,0.5,0.5
                    rows[i][-1,:,:,0]  *= 2

                for j in range(1,L2-2):    self.map[0,j] = ['R']
                self.map[0, 0] = ['SL']
                self.map[0,-2] = ['SR']
                    
                self.ops  = {'SL': sideL, 'SR': sideR, 'R': rows}
                self.conn = {'SL': 'rol', 'SR': 'rol', 'R': 'rol'}
                self.ns   = {'SL': (2,2), 'SR': (2,2), 'R': (2,2)}


            elif  L2 == 2:

                raise(NotImplementedError)
 
                sideI = MPX.copy(bulk_op)           # upper side:
                sideI[0][-1,:,:,1:n1+1] *= 2        # couplings:  1,1,0.5,1
                sideI[1][-1,:,:,1:n1+1] *= 2
                sideI[0][-1,:,:,0] *= 4             # on-site:  1,1,0.5,0.5
                sideI[1][-1,:,:,0] *= 4
                sideI[2][-1,:,:,0] *= 2
                sideI[3][-1,:,:,0] *= 2

                sideO = MPX.copy(bulk_op)           # lower/outer side:
                sideO[1][-1,:,:,1:n1+1] *= 2        # couplings:  0.5,1,1,1
                sideO[3][1:n1+1,:,:, 0] *= 2
                sideO[0][-1,:,:,0] *= 2             # on-site:  0.5,0.5,1,1
                sideO[1][-1,:,:,0] *= 2
                sideO[2][-1,:,:,0] *= 4
                sideO[3][-1,:,:,0] *= 4

                cols = MPX.copy(bulk_op)            # cols:  
                cols[0][ 0,:,:,1:n1+1] *= 2         # couplings:  0.5,1,0.5,1
                cols[1][1:n1+1,:,:, 0] *= 0.5
                cols[1][-1,:,:,1:n1+1] *= 2
                for i in range(4):                  # on-site:    0.5,0.5,0.5,0.5
                    cols[i][-1,:,:,0]  *= 2

                
                for i in range(1,L1-2):    self.map[i,0] = ['C']
                self.map[ 0,0] = ['SI']
                self.map[-2,0] = ['SO']
                    
                self.ops  = {'SI': sideI, 'SO': sideO, 'C': cols}
                self.conn = {'SI': 'rol', 'SO': 'rol', 'C': 'rol'}
                self.ns   = {'SI': (2,2), 'SO': (2,2), 'C': (2,2)}

            else:
                raise(ValueError), 'not valid shape for t1t2 3body model'

        print self.map
        print self.ops.keys()



class t1t2_sum(TrotterOp):
    '''
    sum_<<ij>> Xi Xj + Yi Yj  + gamma*sum_i Xi + tau*sum_i Zi

    as an 4-term MPO x -- x
                          |
                     x -- x

    '''
    def __init__(self,Ls,hs,t1s,t2s):
        
        L1, L2 = Ls

        self.Ls = Ls
        self.db = 2
        self.it_sh = (2,2)

        opList = [paulis['SX'],paulis['SY'],paulis['SZ']]

        if L1 == 1 or L2==1:    # return heisenberg
            heis_pepo  = Heisenberg_sum(Ls,hs,t1s)
            self.ops   = heis_pepo.ops
            self.ns    = heis_pepo.ns
            self.conn  = heis_pepo.conn
            self.map   = heis_pepo.map
            self.it_sh = heis_pepo.it_sh
            self.ind0  = heis_pepo.ind0

        else: 

            '''
            'nn', 'nnn' operators:  first/second set of coupling operators in bottom row/left col
            where a_i, b_i are the 'nn' and 'nnn' coupling terms at the bottom row of matrix i
            and   c_i, d_i are the 'nn' and 'nnn' coupling terms in the left col of matrix i
            we use:
                b1 = t2_13
                b2 = t2_24
 
                a2 = t1_23
                a3 = t1_34
                a1 = t1_14     
		a1*c2 = t1_12 -> c2 = t1_12/t1_14

                d2 = 0
                b3 = anything
                c3 = 1
                d3 = 0
		c4 = 1
                d4 = 0
            '''

            ### "bulk" mpos:  on-sites all have 1/4 weighting, nn couplings have 1/2 weighting
            ###               nnn couplings have 1 weighting
    
            # hs = x, y, z
            hs = np.array(hs)
            onsite = np.zeros((2,2))
            for ind in range(3):         # X, Y, Z  on-site terms
                print ind, hs[ind]
                onsite = onsite + 0.25*hs[ind]*opList[ind].copy()
            # note:  on-site terms:  bulk -- weighting is (0.5,0.5,0.5,0.5)
            #                        corners -- corner site weight is 1.
    
            
            t_inds = np.where(np.abs(np.array(t1s))+np.abs(np.array(t2s)) != 0)[0]   # t1s or t2s != 0
            n1 = len(t_inds) 
            Dw = 2*n1 + 2
            # even when t2=0, need extra space for the long range interaction from 1--4

            # build the actual matrices
            op1 = np.zeros((1 ,2,2,Dw), dtype=np.complex128)
            opm = np.zeros((Dw,2,2,Dw), dtype=np.complex128)
            opL = np.zeros((Dw,2,2,1 ), dtype=np.complex128)
    
    
            # on site terms
            op1[ 0,:,:,0] = onsite.copy()
            opm[-1,:,:,0] = onsite.copy()
            opL[-1,:,:,0] = onsite.copy()

            # nn identities
            op1[ 0,:,:,-1] = np.eye(2) 
            opm[ 0,:,:, 0] = np.eye(2) 
            opm[-1,:,:,-1] = np.eye(2) 
            opL[ 0,:,:, 0] = np.eye(2) 
    
            i1 = 0
            for ti in t_inds:   # t1 couplings

                print ['sx','sy','sz'][ti], t1s[ti]

                x1 = 1+i1
                x2 = 1+n1+i1

                # operator 1
                op1[-1,:,:,x1] = opList[ti]*t1s[ti]*0.5     # operators on bottom row
                op1[-1,:,:,x2] = opList[ti]*t2s[ti]

                # middle operators
                opm[-1,:,:,x1] = opList[ti]*t1s[ti]*0.5     # bottom row
                opm[-1,:,:,x2] = opList[ti]*t2s[ti]
                opm[x2,:,:,x1] = np.eye(2)                  # long range identities
                opm[x1,:,:,x2] = np.eye(2)
                opm[x1,:,:, 0] = opList[ti]                 # operators on left col

                # operator 4 (last)
                opL[x1,:,:, 0] = opList[ti]                 

                i1 += 1      

 
    
            print 'bulk op'
            bulk_op = MPX.MPX( [op1,opm.copy(),opm.copy(),opL], phys_bonds=[(2,2)]*4 )
            print len(bulk_op), [m.shape for m in bulk_op]

            # print Dw
            # for i in range(Dw):
            #     print op1[0,:,:,i]

            ### border/exception mpos:
            self.map = np.empty((L1,L2),dtype=list)
            for idx in np.ndindex((L1,L2)):   self.map[idx] = ['B']

            for i in range(L1):   self.map[i][-1] = []    # set right col, bottom row --> empty lists (since ns = 2,
            for j in range(L2):   self.map[-1][j] = []    # no trotter steps applied at these lattice sites

    
            if L1 > 2 and L2 > 2:   
                cornerLI = MPX.copy(bulk_op)        # upper left corner
                cornerLI[0][0,:,:,1:n1+1] *= 2      # couplings:  1,0.5,0.5,1
                cornerLI[0][-1,:,:,0] *= 4          # on-site:    1,0.5,0.25,0.5
                cornerLI[1][-1,:,:,0] *= 2
                cornerLI[3][-1,:,:,0] *= 2
      
                rowI = MPX.copy(bulk_op)            # upper row:
                rowI[1][1:n1+1,:,:,0] *= 2          # couplings:  1,0.5,0.5,0.5
                rowI[0][-1,:,:,0] *= 2              # on-site:    0.5,0.5,0.25,0.25
                rowI[1][-1,:,:,0] *= 2

                cornerIR = MPX.copy(bulk_op)        # upper right corner: 
                cornerIR[1][1:n1+1,:,:, 0] *= 2     # couplings:  1,1,0.5,0.5   (1<->2)
                cornerIR[1][-1,:,:,1:n1+1] *= 2     # (2<->3)
                cornerIR[0][-1,:,:,0] *= 2          # on-site:    0.5,1,0.5,0.25
                cornerIR[1][-1,:,:,0] *= 4
                cornerIR[2][-1,:,:,0] *= 2
                
                colR = MPX.copy(bulk_op)            # right col:  
                colR[1][-1,:,:,1:n1+1] *= 2         # couplings:  0.5,1,0.5,0.5
                colR[1][-1,:,:,0] *= 2              # on-site:    0.25,0.5,0.5,0.25
                colR[2][-1,:,:,0] *= 2

                cornerOR = MPX.copy(bulk_op)        # lower right corner:
                cornerOR[1][-1,:,:,1:n1+1] *= 2     # couplings:  0.5,1,1,0.5   (2<-3)
                cornerOR[2][-1,:,:,1:n1+1] *= 2     # (3<->4)
                cornerOR[1][-1,:,:,0] *= 2          # on-site:    0.25,0.5,1,0.5
                cornerOR[2][-1,:,:,0] *= 4
                cornerOR[3][-1,:,:,0] *= 2
                
                rowO = MPX.copy(bulk_op)            # lower row"
                rowO[2][-1,:,:,1:n1+1] *= 2         # couplings:  0.5,0.5,1,0.5
                rowO[2][-1,:,:,0] *= 2              # on-site:    0.25,0.25,0.5,0.5
                rowO[3][-1,:,:,0] *= 2

                cornerLO = MPX.copy(bulk_op)        # lower left corner:
                cornerLO[0][-1,:,:,1:n1+1] *= 2     # couplings:  0.5,0.5,1,1 
                cornerLO[1][ 1:n1+1,:,:,0] *= 0.5
                cornerLO[2][-1,:,:,1:n1+1] *= 2      
                cornerLO[0][-1,:,:,0] *= 2          # on-site:    0.5,0.25,0.5,1
                cornerLO[2][-1,:,:,0] *= 2
                cornerLO[3][-1,:,:,0] *= 4

                colL = MPX.copy(bulk_op)            # left col
                colL[0][ 0,:,:,1:n1+1] *= 2         # couplings:  0.5,0.5,0.5,1
                colL[1][ 1:n1+1,:,:,0] *= 0.5
                colL[0][-1,:,:,0] *= 2              # on-site:    0.5,0.25,0.25,0.5
                colL[3][-1,:,:,0] *= 2

                for i in range(1,L1-2): 
                    self.map[i, 0] = ['LC']   # left column
                    self.map[i,-2] = ['RC']   # right column

                for j in range(1,L2-2):
                    self.map[ 0,j] = ['IR']   # inner/upper row
                    self.map[-2,j] = ['OR']   # outer/lower row

                self.map[ 0, 0] = ['LIC']
                self.map[-2, 0] = ['LOC']
                self.map[-2,-2] = ['ROC']
                self.map[ 0,-2] = ['RIC']

                for i in range(L1):    self.map[i,-1] = []
                for j in range(L2):    self.map[-1,j] = []


                self.ops  = {'B'  : bulk_op,  'LIC': cornerLI, 'IR' : rowI,     'RIC': cornerIR, 'RC' : colR,
                             'ROC': cornerOR, 'OR' : rowO,     'LOC': cornerLO, 'LC' : colL}
                # conn_list = ['rol','oli','lir','iro']
                self.conn = {'B'  : 'rol', 'LIC': 'rol', 'IR' : 'rol', 'RIC': 'rol', 'RC' : 'rol',
                             'ROC': 'rol', 'OR' : 'rol', 'LOC': 'rol', 'LC' : 'rol'}
                self.ns   = {'B'  : (2,2), 'LIC': (2,2), 'IR' : (2,2), 'RIC': (2,2), 'RC' : (2,2),
                             'ROC': (2,2), 'OR' : (2,2), 'LOC': (2,2), 'LC' : (2,2)}
                self.ind0 = {'B'  : (0,0), 'LIC': (0,0), 'IR' : (0,0), 'RIC': (0,0), 'RC' : (0,0),
                             'ROC': (0,0), 'OR' : (0,0), 'LOC': (0,0), 'LC' : (0,0)}

                # print 'heis loop', self.map
                # print 'bulk', [m.transpose(0,3,1,2) for m in self.ops['B']]
                # print 'right col', [m.transpose(0,3,1,2) for m in self.ops['RC']]
                # print 'left col', [m.transpose(0,3,1,2) for m in self.ops['LC']]
                # print 'inner row', [m.transpose(0,3,1,2) for m in self.ops['IR']]
                # print 'outer row', [m.transpose(0,3,1,2) for m in self.ops['OR']]

            elif L1 == 2 and L2 == 2:    # bulk mpo --> boundary mpo
                
                new_op = bulk_op.copy()
                for i in range(4):       new_op[i][-1,:,:,0] *= 4        # on-site
                for i in range(3):       new_op[i][-1,:,:,1:n1+1] *= 2   # couplings

                self.map[0,0] = ['B']
                self.ops  = {'B': new_op}
                self.conn = {'B': 'rol'}
                self.ns   = {'B': (2,2)}
                self.ind0 = {'B': (0,0)}

                # print 't1-t2'
                # ind = 0
                # for m in new_op:
                #     for(i,j) in np.ndindex(m.shape[0],m.shape[-1]):
                #         print ind,i,j, m[i,:,:,j]
                #     ind += 1

                # print self.map
 
            elif L1 == 2:
 
                sideL = MPX.copy(bulk_op)           # left side
                sideL[0][ 0,:,:,1:n1+1] *= 2        # couplings:  1,0.5,1,1
                sideL[2][-1,:,:,1:n1+1] *= 2
                sideL[0][-1,:,:,0] *= 4             # on-site:  1,0.5,0.5,1
                sideL[1][-1,:,:,0] *= 2
                sideL[2][-1,:,:,0] *= 2
                sideL[3][-1,:,:,0] *= 4
                
                sideR = MPX.copy(bulk_op)           # right side
                sideR[1][1:n1+1,:,:, 0] *= 2        # couplings:  1,1,1,0.5
                sideR[1][-1,:,:,1:n1+1] *= 2
                sideR[2][-1,:,:,1:n1+1] *= 2
                sideR[0][-1,:,:,0] *= 2             # on-site:  0.5,1,1,0.5
                sideR[1][-1,:,:,0] *= 4
                sideR[2][-1,:,:,0] *= 4
                sideR[3][-1,:,:,0] *= 2

                rows = MPX.copy(bulk_op)            # 2 rows
                rows[1][1:n1+1,:,:, 0] *= 2         # couplings:  1,0.5,1,0.5
                rows[2][-1,:,:,1:n1+1] *= 2
                for i in range(4):                  # on-site:    0.5,0.5,0.5,0.5
                    rows[i][-1,:,:,0]  *= 2

                for j in range(1,L2-2):    self.map[0,j] = ['R']
                self.map[0, 0] = ['SL']
                self.map[0,-2] = ['SR']
                    
                self.ops  = {'SL': sideL, 'SR': sideR, 'R': rows}
                self.conn = {'SL': 'rol', 'SR': 'rol', 'R': 'rol'}
                self.ns   = {'SL': (2,2), 'SR': (2,2), 'R': (2,2)}
                self.ind0 = {'SL': (0,0), 'SR': (0,0), 'R': (0,0)}

                # print sideL[0].shape
                # print sideL[0][-1,:,:,0]
                # for i in range(Dw):
                #     print 'sideL 0', i, sideL[0][0,:,:,i]

                # for i in range(Dw):
                #     for j in range(Dw):
                #         print 'sideL 1', i,j, sideL[1][i,:,:,j]

                # for i in range(Dw):
                #     print 'sideL 3',i, sideL[3][i,:,:,0]


            elif  L2 == 2:

                sideI = MPX.copy(bulk_op)           # upper side:
                sideI[0][-1,:,:,1:n1+1] *= 2        # couplings:  1,1,0.5,1
                sideI[1][-1,:,:,1:n1+1] *= 2
                sideI[0][-1,:,:,0] *= 4             # on-site:  1,1,0.5,0.5
                sideI[1][-1,:,:,0] *= 4
                sideI[2][-1,:,:,0] *= 2
                sideI[3][-1,:,:,0] *= 2

                sideO = MPX.copy(bulk_op)           # lower/outer side:
                sideO[1][-1,:,:,1:n1+1] *= 2        # couplings:  0.5,1,1,1
                sideO[3][1:n1+1,:,:, 0] *= 2
                sideO[0][-1,:,:,0] *= 2             # on-site:  0.5,0.5,1,1
                sideO[1][-1,:,:,0] *= 2
                sideO[2][-1,:,:,0] *= 4
                sideO[3][-1,:,:,0] *= 4

                cols = MPX.copy(bulk_op)            # cols:  
                cols[0][ 0,:,:,1:n1+1] *= 2         # couplings:  0.5,1,0.5,1
                cols[1][1:n1+1,:,:, 0] *= 0.5
                cols[1][-1,:,:,1:n1+1] *= 2
                for i in range(4):                  # on-site:    0.5,0.5,0.5,0.5
                    cols[i][-1,:,:,0]  *= 2

                
                for i in range(1,L1-2):    self.map[i,0] = ['C']
                self.map[ 0,0] = ['SI']
                self.map[-2,0] = ['SO']
                    
                self.ops  = {'SI': sideI, 'SO': sideO, 'C': cols}
                self.conn = {'SI': 'rol', 'SO': 'rol', 'C': 'rol'}
                self.ns   = {'SI': (2,2), 'SO': (2,2), 'C': (2,2)}
                self.ind0 = {'SI': (0,0), 'SO': (0,0), 'C': (0,0)}




#### class definitions ####
# MPO defintion
class PEPO:
    def __init__(self,Ls):
        
        self.mpo_type  = None           # ising, hard-core boson...
        
        self.Ws = {}
        self.ds = [2]
        self.Dw = 1
        self.Ls = Ls
        
        self.sites = [None for i in range(L)]   
        # list of keys corresponding to MPO at that site
        # None: either the identity or previous site has multiple site block which operates on it
  

    def get(self,ind):
        # returns MPO associated with lattice site ind = (i,j)
        
        i,j = ind
        return self.Ws[self.sites[i,j]]

    def getList(self):
        # returns full PEPO list
        L1, L2 = self.Ls
        mpoList = [[self.get((i,j)) for j in range(L2)] for i in range(L1)]
        return mpoList

    # def getSites(self,ind0,numSites):
    #     # returns ndarray
    #     tenslist = self.getList()
    #     block = tenslist[ind0]
    #     for i in range(1,numSites):
    #         block = np.tensordot(block, tenslist[i], axes=(-1,0))
    #     return block

    def getPEPO(self):
        mpoList = self.getList()
        return PEPX.PEPX(mpoList)


    def update(self,pos,blockname,newMPO=None):
        # pos = lattice position
        
        try:
             newMPO = self.Ws[blockname]
             self.sites[pos-1] = blockname
        except (KeyError):
             self.Ws[blockname] = newMPO
             self.sites[pos-1]  = blockname

        self.Dw = max(self.Dw,np.max(newMPO.shape))


    def scale(self,fac):    # scales MPO by a constant |fac|

        notNone = np.where(self.sites is not None)[0]
        nMats   = np.sum(notNone)
        c = abs(fac)**(1./float(nMats))
        Wkeys = self.Ws.keys()
        for wk in WKeys:
           oldBlock = self.Ws[wk]
           self.Ws[wk] = c*oldBlock


    def maxEVal(self):
        """
        returns minimum and maximum eigenvals (for time step convergence estimates)

        """
        import scipy.linalg as LA

        MPOlist = self.getList()
        ns = self.L

        mat = MPOlist[0]
        for m in MPOlist[1:]:
            mat = np.einsum('l...abr,rcds->l...abcds',mat,m)

        axT = [0] + np.arange(1,2*ns+1).reshape(ns,2).transpose().reshape(-1).tolist() + [-1]
        mat = mat.transpose(axT)
        
        neig = np.prod(self.ds)
        mat = mat.reshape(neig,neig)

        max_e = LA.eigh(mat,eigvals=(neig-1,neig-1),eigvals_only=True)
        print max_e
        min_e = LA.eigh(mat,eigvals=(0,0),eigvals_only=True)
        print min_e

        return min_e[0], max_e[0]
        
        
    def maxEVal_dmrg(self,DMAX=20,conv_err=1.0e-3):
        """
        returns minimum and maximum eigenvals (for time step convergence estimates)

        """
        import MPX_GSEsolver

        H_MPO = self.getMPX()
        max_evec, max_eval = MPX_GSEsolver.solver(H_MPO,target=-1,DMAX=DMAX, conv_err=conv_err)

        return max_eval
        
        

class PEPO_Heisenberg(PEPO):
    def __init__(self,Ls,hs,Js):
        
        self.Ls = Ls
        self.hs = hs
        self.Js = Js

        ops = paulis
        ops['SP'] = sp
        ops['SM'] = sm

        # hs = x, y, z
        hs = np.array(hs)
        singList = ['SX','SY','SZ']
        # Js = xx, yy, zz, xy, yz, zx
        Js = np.array(Js)
        # coupList = [('SX','SX'),('SY','SY'),('SZ','SZ'),('SX','SY'),('SY','SZ'),('SZ','SX')]
        coupList = [('SP','SM'),('SM','SP'),('SZ','SZ'),('SX','SY'),('SY','SZ'),('SZ','SX')]
        ## the two are equivalent
        
        if len(Js) <= 3:
            Js1 = np.where(Js != 0)[0]
            Js2 = []
            Dw = len(Js1) + 2
        else:
            Js1 = np.where(np.array(Js[:3]) != 0)[0]
            Js2 = np.where(np.array(Js[3:]) != 0)[0]
            Dw = len(Js1) + len(Js2)*2 + 2
            
        W = np.zeros((Dw,2,2,Dw),dtype=np.complex128)
        
        # W
        W[0,:,:,0]   = ops['ID']
        W[-1,:,:,-1] = ops['ID']
        
        for ind in range(3):
            if not hs[ind] == 0:
                W[-1,:,:,0] = W[-1,:,:,0].copy() + hs[ind]*ops[singList[ind]].copy()
        
        for ii in range(len(Js1)):
            ind = Js1[ii]
            W[1+ii,:,:,0]  = ops[coupList[ind][0]].copy()
            W[-1,:,:,1+ii] = ops[coupList[ind][1]].copy()*Js[ind]
            
        for ii in range(len(Js2)):
            ind = Js2[ii]+3
            W[1+len(Js1)+2*ii,:,:,0]   = ops[coupList[ind][0]].copy()
            W[-1,:,:,1+len(Js1)+2*ii]  = ops[coupList[ind][1]].copy()*Js[ind]
            W[1+len(Js1)+2*ii+1,:,:,0] = ops[coupList[ind][1]].copy()
            W[-1,:,:,1+len(Js1)+2*ii]  = ops[coupList[ind][0]].copy()*Js[ind]
                    
        
        W1 = W[-1:,:,:,:]
        WL = W[:,:,:,:1]
        
        self.Dw = Dw
        self.L  = L
        self.ds = [2]*L
        self.hs = hs
        self.Js = Js

        self.Ws = {'W1': W1.copy(), 'W': W.copy(), 'WL': WL.copy()}
        self.sites = ['W1'] + ['W' for i in range(L-2)] + ['WL']
        self.ops = ops

        ### not yet implemented


#  temp: can easily be replaced by heisenberg

class PEPO_mag(PEPO):
    def __init__(self,Ls,pauli_op):

        self.Ls = Ls
        self.ds = np.ones(Ls)*2
        self.Dw = 2

        Dw = 2
        L1, L2 = Ls

        if L1 == 1 and L2 == 1:
            Wb = paulis[pauli_op].reshape(1,1,1,1,2,2)
            self.sites = np.array([['Wb']],dtype='S3')
            self.Ws = {'Wb': Wb.copy()}

        elif L2 == 1:   # lattice is a single column
            # snake vertically across PEPS
            # bonds are liorud
            W0O = np.zeros((1,1,Dw,1,2,2))      # left end
            WIO = np.zeros((1,Dw,Dw,1,2,2))     # horizontal left to right
            WI0 = np.zeros((1,Dw,1,1,2,2))      # right end
    
            W0O[0,0,0,0,:,:] = paulis[pauli_op]
            W0O[0,0,1,0,:,:] = np.eye(2)
    
    	    WIO[0,0,0,0,:,:] = np.eye(2)
            WIO[0,1,0,0,:,:] = paulis[pauli_op]
            WIO[0,1,1,0,:,:] = np.eye(2)
    
            WI0[0,0,0,0,:,:] = np.eye(2)
            WI0[0,1,0,0,:,:] = paulis[pauli_op]
    
            self.sites = np.empty((L1,L2),dtype='S3')
            for idx in np.ndindex((L1,L2)):
                self.sites[idx] = 'WIO'
            self.sites[ 0,0] = 'W0O'
            self.sites[-1,0] = 'WI0'

            self.Ws = {'WIO': WIO.copy(), 'W0O': W0O.copy(), 'WI0': WI0.copy()}
        
        else:
            # snake across PEPS to avoid double counting
            # bonds are liorud
            W0R = np.zeros((1,1,1,Dw,2,2))      # left end
            WLR = np.zeros((Dw,1,1,Dw,2,2))     # horizontal left to right
            WLO = np.zeros((Dw,1,Dw,1,2,2))     # left bond to outer/down
            WIL = np.zeros((Dw,Dw,1,1,2,2))     # outer/down to right bond
            WRO = np.zeros((1,1,Dw,Dw,2,2))     # right to left bond
            WIR = np.zeros((1,Dw,1,Dw,2,2))     # right to outer bond
            WL0 = np.zeros((Dw,1,1,1,2,2))      # right end
    
            W0R[0,0,0,0,:,:] = paulis[pauli_op]
            W0R[0,0,0,1,:,:] = np.eye(2)
    
    	    WLR[0,0,0,0,:,:] = np.eye(2)
            WLR[1,0,0,0,:,:] = paulis[pauli_op]
            WLR[1,0,0,1,:,:] = np.eye(2)
    
            WLO[0,0,1,0,:,:] = np.eye(2)       # does a transpose (from row i to row i+1)
            WLO[1,0,1,0,:,:] = paulis[pauli_op]
            WLO[1,0,0,0,:,:] = np.eye(2)
    
            WIL[0,0,0,0,:,:] = np.eye(2)
            WIL[1,0,0,0,:,:] = paulis[pauli_op]
            WIL[1,1,0,0,:,:] = np.eye(2)
    
            WRO[0,0,0,1,:,:] = np.eye(2)       # does a transpose (from row i+1 to row i+2)
            WRO[0,0,0,0,:,:] = paulis[pauli_op]
            WRO[0,0,1,0,:,:] = np.eye(2)
    
            WIR[0,0,0,0,:,:] = np.eye(2)
            WIR[0,1,0,0,:,:] = paulis[pauli_op]
            WIR[0,1,0,1,:,:] = np.eye(2)
    
            WL0[0,0,0,0,:,:] = np.eye(2)
            WL0[1,0,0,0,:,:] = paulis[pauli_op]
    
            self.sites = np.empty((L1,L2),dtype='S3')
            for idx in np.ndindex((L1,L2)):
                self.sites[idx] = 'WLR'
            for i in range(0,L1,2):
                self.sites[i,0]  = 'WIR'
                self.sites[i,-1] = 'WLO'
            for i in range(1,L1,2):
                self.sites[i,-1] = 'WIL'
                self.sites[i, 0] = 'WRO'
    
            self.sites[0,0] = 'W0R'
            # if L1 == 1:      self.sites[ 0,-1] = 'WL0'
            if L1%2 == 1:    self.sites[-1,-1] = 'WL0'
            else:            self.sites[-1, 0] = 'W0R'
    
    
            self.Ws = {'WLR': WLR.copy(), 'WIR': WIR.copy(), 'WLO': WLO.copy(), 'WIL': WIL.copy(), 
                       'WRO': WRO.copy(), 'W0R': W0R.copy(), 'WL0': WL0.copy()}

        

