import sys
import time
from io import StringIO
import numpy as np

import tens_fcts as tf
import MPX
import MPX_GSEsolver




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



def add(block1,block2,ends=(False,False)):
    DL1,d1,d2,DR1 = block1.shape
    DL2,DR2       = block2.shape[0],block2.shape[-1]
    
    # # assert(block1.shape[1:3] == block2.shape[1:3]), block1.shape, block2.shape
    # if not block1.shape[1:3] == block2.shape[1:3]:
    #     print 'error in Op.add'
    #     print block1.shape, block2.shape
    #     exit()
    
    newBlock = np.zeros((DL1+DL2,d1,d2,DR1+DR2),dtype=np.complex128)
    newBlock[:DL1,:,:,:DR1] = block1.copy()
    newBlock[DL1:,:,:,DR1:] = block2.copy()
    
    if ends[0]:     # MPO is at the left end  --> sum vertically so 1 x (DR1+DR2) x d x d
        newBlock = np.einsum('ludr->udr',newBlock)
        # print newBlock.shape
        newBlock = newBlock.reshape(1,d1,d2,DR1+DR2)
    
    if ends[1]:     # MPO is at the right end --> sum horizontally so (DL1+DL2) x 1 x d x d
        newBlock = np.einsum('ludr->lud',newBlock)
        m = newBlock.shape[0]
        newBlock = newBlock.reshape(m,d1,d2,1)
        
    return newBlock


def prod(block1,block2):
    # bottom MPO, topMPO'
    newBlock = np.einsum('aijb,cjkd->acikbd',block2,block1)
    dimL  = block1.shape[0] *block2.shape[0]
    dimR  = block1.shape[-1]*block2.shape[-1]
    dimU,dimD = block1.shape[1:3]
    newBlock = newBlock.reshape(dimL,dimU,dimD,dimR)
    return newBlock


def decomposeMPO(MPOblock,nSites,order='io'):
    # MPOblock:  DwL x (all out, up) x (all in, down) x DwR
    
    # reshape to DwL x (out/in)*nSites x DwR. then use decomposeM in state fct
    if order == 'io':
        d_site = np.arange(1,1+2*nSites).reshape(2,nSites).transpose()  # labels for middle bonds
        d_orig = np.array(MPOblock.shape)[d_site]
        d_eff = np.prod(d_orig,1)
        
        d_site = d_site.reshape(-1)        
        
        assert(len(d_eff) == nSites)
        
        DwL,DwR = MPOblock.shape[0], MPOblock.shape[-1]
        MPSform = MPOblock.transpose([0]+d_site.tolist()+[-1])     # list = axes reordered to this order
        MPSform = MPSform.reshape([DwL]+d_eff.tolist()+[DwR])
        
    elif order == 'site':       # already sorted by site
        d_site = np.arange(1,2*nSites+1).reshape(nSites,2)  # labels for middle bonds
        d_orig = np.array(MPOblock.shape)[d_site]
        d_eff  = np.prod(d_orig,1)
        assert(len(d_eff) == nSites)
    
        DwL,DwR = MPOblock.shape[0],MPOblock.shape[-1]
        MPSform = MPOblock.reshape([DwL]+d_eff.tolist()+[DwR])
    elif order == 'siteMPS':    # already in MPS form
        DwL,DwR = MPOblock.shape[0],MPOblock.shape[-1]
        MPSform = MPOblock.copy()
    else:
        print('please provide valid order, not ', order)
        
    
    bL = np.ones(DwL)#/np.sqrt(DwL)
    bR = np.ones(DwR)#/np.sqrt(DwR)
    bonds,sites,errs = St.decomposeM(MPSform,bL,bR)
    
        
    newMPOs = []
    for i in range(len(sites)):
        if i == nSites - 1:     mpo = sites[i].copy()       # last site
        else:                   mpo = tf.dMult('MD',sites[i],bonds[i])
        if order == 'io' or order == 'site':      # reshape to MPO form
            m,d_eff,n = mpo.shape
            d1,d2 = d_orig[i]
            assert(int(d_eff) == int(d1*d2)), '%d %d %d'%(d_eff,d1,d2)
            
            mpo = mpo.reshape(m,d1,d2,n)
        newMPOs.append(mpo)   
    
    return newMPOs,errs
    

def compressMPO(MPOlist):
    # MPOlist = list of MPO blocks that make up a full MPO
    
    newMPO  = MPOlist[:]
    newErrs = [0.0 for ii in range(len(MPOlist)-1)]

    for i in range(len(newMPO)-1):
        m1,m2 = newMPO[i:i+2]
        m1_os = m1.shape[:-1]
        m2_os = m2.shape[1:]
	m1_ns = (m1_os[0],int(np.prod(m1_os[1:])),m1.shape[-1])
        m2_ns = (m2.shape[0],int(np.prod(m2_os[:-1])),m2_os[-1])

        mpo_2site = np.einsum('iaj,jbk->iabk',m1.reshape(m1_ns),m2.reshape(m2_ns))   #DwL x out (up) x in (down) x DwR
        mpos,compErrs = decomposeMPO(mpo_2site,2,'siteMPS')
        mpo1,mpo2 = mpos
        newMPO[i]   = mpo1.reshape(m1_os+(mpo1.shape[-1],))
        newMPO[i+1] = mpo2.reshape((mpo2.shape[0],)+m2_os)
        newErrs[i] = compErrs[0]

    return newMPO,newErrs


def mpo_EVals(mpo):
    """
    returns minimum and maximum eigenvals (for time step convergence estimates)

    """
    import scipy.linalg as LA

    L = len(mpo)
    mpo_block = MPX.getSites(mpo,0,L)
    # print 'time_block'
    # for s in mpo:  print s.transpose(0,3,1,2), '\n'
    axT = tf.site2io(L)
    mpo_block = mpo_block.transpose(axT)

    t_sh = mpo_block.shape
    sqdim = int(np.sqrt(np.prod(t_sh)))
    H = mpo_block.reshape(sqdim,sqdim)

    evals = LA.eigvals(H)

    return evals


def mpo_targetEVal(H_MPO,target=-1,DMAX=20,conv_err=1.e-3):
    '''
    determine 0.95*maximum stable tau for RK4 time evolution
    '''
    max_evec, max_eval = MPX_GSEsolver.solver(H_MPO,target,DMAX, conv_err)

    return max_eval


    

#### class definitions ####
class MPO:
    def __init__(self,L):
        
        self.mpo_type  = None           # ising, hard-core boson...
        
        self.Ws = {}
        self.ds = [2]
        self.Dw = 1
        self.L  = L
        
        self.sites = [None for i in range(L)]   
        # list of keys corresponding to MPO at that site
        # None: either the identity or previous site has multiple site block which operates on it
  

    def copy(self):
        MPOcopy       = MPO(self.L)
        MPOcopy.Ws    = self.Ws   # to copy:  self.Ws.copy()
        MPOcopy.Dw    = self.Dw
        MPOcopy.sites = self.sites[:]
        return MPOcopy

            
    def get(self,i):
        # returns MPO associated with lattice site i
        # if i is zero, returns identity
        
        if self.sites[i] is None:
            return None
        else:
            return self.Ws[self.sites[i]]

    def getList(self):
        # returns full MPO list
        mpoList = [self.get(i) for i in range(self.L)]
        return mpoList

    def getSites(self,ind0,numSites):
        # returns ndarray
        tenslist = self.getList()
        block = tenslist[ind0]
        for i in range(1,numSites):
            block = np.tensordot(block, tenslist[i], axes=(-1,0))
        return block

    def getMPX(self):
        mpoList = self.getList()
        return MPX.MPX(mpoList)


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
        
        

class MPO_Heisenberg(MPO):
    def __init__(self,L,hs,Js):
        self.mpo_type = 'Heisenberg'
        
        #### spin model operators ####
        ops = paulis
        # ops['ID'] = np.array([[1.,0],[0,1.]])
        # ops['SX'] = np.array([[0,1.],[1.,0]])
        # ops['SY'] = np.array([[0,-1.j],[1.j,0]])
        # ops['SZ'] = np.array([[1.,0],[0,-1.]])


        # hs = x, y, z
        hs = np.array(hs)
        singList = ['SX','SY','SZ']
        # Js = xx, yy, zz, xy, yz, zx
        Js = np.array(Js)
        coupList = [('SX','SX'),('SY','SY'),('SZ','SZ'),('SX','SY'),('SY','SZ'),('SZ','SX')]
        
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

        self.Ws = {'W1': W1.copy(), 'W': W.copy(), 'WL': WL.copy()}
        self.sites = ['W1'] + ['W' for i in range(L-2)] + ['WL']
        self.ops = ops

        

        
class MPO_impurity(MPO):     # occupation of each site:  0 or 1

    ## spinless Anderson model:
    
    ## Anderson 1:  impurity is at the first site, coupled to 1D environement
    ## H =
    ##   impurity:       \sum_q       eq       q+q         (no coulombic repulsion)
    ##   environment:  + \sum_rj,r'j' trj,r'j' rj+r'j' + cc
    ##   coupling:     + \sum_q,rj    Vq,rj    q+rj    + cc
    ## (q,r = occupation #s, j = lattice site position)
    
    ## Anderson 2:  impurity coupled to multiple environment sites?
    
    def __init__(self,L,model,coeffs):

        #### Hard-core boson operators  ####
        ops = {}
        ops['b+'] = np.array([[0,0],[1.,0]])
        ops['b-'] = np.array([[0,1.],[0,0]])
        ops['n']  = np.array([[0,0],[0,1.]])
        ops['ID'] = np.eye(2) 
        
        self.ds = [2]*L
        self.L = L
        if model == 'hcboson':
            e_imp, t_self, t_nn, v = coeffs 
        
            self.d = 2
            self.Dw = 4
            self.mpo_type = 'hcboson'
            
            W1 = np.zeros((1,2,2,self.Dw))
            We = np.zeros((self.Dw,2,2,self.Dw))
            
            ## impurity Hamiltonian
            W1[0,:,:,0] = e_imp*ops['n']
            W1[0,:,:,1] = v*ops['b+']
            W1[0,:,:,2] = np.conj(v)*ops['b-']
            W1[0,:,:,3] = ops['ID']
            
            We[0,:,:,0] = ops['ID']
            We[1,:,:,0] = ops['b-']
            We[2,:,:,0] = ops['b+']
            We[3,:,:,0] = t_self*ops['n']
            We[3,:,:,1] = t_nn*ops['b+']
            We[3,:,:,2] = np.conj(t_nn)*ops['b-']
            We[3,:,:,3] = ops['ID']
    
            WL = We[:,:,:,:1].copy()

            self.Ws = {'W1':W1.copy(),'W':We.copy(),'WL': WL.copy()}
            self.sites = ['W1'] + ['W' for i in range(L-2)] + ['WL']
            self.ops = ops

            # can get back site,interaction,and bath MPOs:
            # site:      W1[0,0]
            # coupling:  W1[0,1:2],We[1:2,0]
            # bath:      We[2:,:], We, ...
           

        else:
            print('not yet implemented')
        


    def getComponents(self):
        # break Hamiltonian into site/impurity, interaction, bath terms
        W1 = self.Ws['W1']
        We = self.Ws['W' ]
        WL = self.Ws['WL']

        # returns mini MPOs
        Wsite = MPO(self.L)
        Wsite.update(1,'W1', W1[:1,:,:,:1].copy())

        Wint  = MPO(self.L)
        Wint.update(1,'Wv1', W1[:1,:,:,1:-1].copy())
        Wint.update(2,'Wv2', We[1:-1,:,:,:1].copy())

        Wbath = MPO(self.L)
        Wbath.update(2,'W1', We[2:,:,:,:].copy())
        for ind in range(3,self.L):
            Wbath.update(ind,'W',We.copy())
        Wbath.update(self.L,'WL',WL.copy())

        return Wsite, Wint, Wbath        


        
class MPO_spinboson(MPO):
    ## in 1D MPS geometry, but with long-range interactions
        
    ## spin boson hamiltonian
    ## H =  
    ##   spin:           \sum_q    e0 SZ  ( + U  u+ud+d)
    ##   environment:  + \sum_j,j' ej bj+bj
    ##   coupling:     + \sum_q,j  vk SZ(bj+ + bj)
    ## (q = spin impurity, j = lattice site position)
     
    
    def __init__(self,n_bath,d_boson, e0, t0, eks, vks):
        self.ops = {}
        self.ops['b+'] = np.diag([np.sqrt(x) for x in range(1,d_boson)],k=-1)  # -1 below diagonal
        self.ops['b-'] = np.diag([np.sqrt(x) for x in range(1,d_boson)],k= 1)  #  1 above diagonal
        self.ops['bn'] = np.diag(range(d_boson))*1.0
        self.ops['bi'] = np.diag([1.0]*d_boson)*1.0

        self.ops['SX'] = np.array([[0.,1.],[1.,0.]])
        self.ops['SZ'] = np.array([[1.,0.],[0.,-1.]])

        self.Dw = 3
        df      = 2
        db      = d_boson
        self.ds = [df] + [db]*n_bath
        self.eks = eks
        self.vks = vks
        self.e0  = e0
        self.t0  = t0

        self.L = n_bath + 1


        # #### prior to 12/4/18 ####
        # ## MPO on spin site
        # WS = np.zeros([1,df,df,self.Dw])
        # WS[0,:,:,0] = -(t0/2)*self.ops['SX'] - (e0/2)*self.ops['SZ']      # tunneling amplitude + site energy
        # WS[0,:,:,1] = self.ops['SZ']/2./np.sqrt(np.pi)                    # creation of fermion
        # WS[0,:,:,2] = np.eye(df)

        # self.Ws = {'WS': WS.copy()}
        # self.sites = ['WS']

        # ## MPo on bath sites
        # W  = np.zeros([self.Dw,db,db,self.Dw])
        # W[0,:,:,0] = np.eye(db)
        # W[1,:,:,1] = np.eye(db)                 # propagates SZ interaction term
        # W[2,:,:,2] = np.eye(db)

        # WL = np.zeros([self.Dw,db,db,1])
        # WL[0,:,:,0] = np.eye(db)

        # for i in range(n_bath-1):            
        #     W[1,:,:,0] = vks[i]*(self.ops['b-'] + self.ops['b+']) # interaction with spin
        #     # W[1,:,:,0] = vks[i]/2./np.sqrt(np.pi)*self.ops['b-'] \
        #     #              + np.conj(vks[i])/2./np.sqrt(np.pi)*self.ops['b+'] # interaction with spin
        #     W[2,:,:,0] = eks[i]*self.ops['bn']  # energy of boson 
        #     W_name = 'W'+str(i)
        #     self.Ws[W_name] = W.copy()
        #     self.sites += [W_name]

        # # WL[1,:,:,0] = vks[-1]/2./np.sqrt(np.pi)*self.ops['b-'] \
        # #                + np.conj(vks[-1])/2./np.sqrt(np.pi)*self.ops['b+'] # interaction with spin
        # WL[1,:,:,0] = vks[-1]*(self.ops['b-'] + self.ops['b+']) # interaction with spin
        # WL[2,:,:,0] = eks[-1]*self.ops['bn']    # energy of boson
        # self.Ws['WL'] = WL.copy()
        # self.sites += ['WL']


        #### post 12/4/18:  following Wang + Thoss 2008 ####
        ## MPO on spin site
        WS = np.zeros([1,df,df,self.Dw])
        WS[0,:,:,0] = -(t0)*self.ops['SX'] - (e0)*self.ops['SZ']      # tunneling amplitude + site energy
        WS[0,:,:,1] = self.ops['SZ']                                  # creation of fermion
        WS[0,:,:,2] = np.eye(df)

        self.Ws = {'WS': WS.copy()}
        self.sites = ['WS']

        ## MPo on bath sites
        W  = np.zeros([self.Dw,db,db,self.Dw])
        W[0,:,:,0] = np.eye(db)
        W[1,:,:,1] = np.eye(db)                 # propagates SZ interaction term
        W[2,:,:,2] = np.eye(db)

        WL = np.zeros([self.Dw,db,db,1])
        WL[0,:,:,0] = np.eye(db)

        for i in range(n_bath-1):            
            W[1,:,:,0] = vks[i]*(self.ops['b-'] + self.ops['b+']    )  # interaction with spin
            # W[2,:,:,0] = eks[i]*(self.ops['bn'] + self.ops['bi']*0.5)  # energy of boson 
            W[2,:,:,0] = eks[i]*(self.ops['bn'])  # energy of boson w/o offset
            W_name = 'W'+str(i)
            self.Ws[W_name] = W.copy()
            self.sites += [W_name]

        WL[1,:,:,0] = vks[-1]*(self.ops['b-'] + self.ops['b+']    )  # interaction with spin
        # WL[2,:,:,0] = eks[-1]*(self.ops['bn'] + self.ops['bi']*0.5)  # energy of boson
        WL[2,:,:,0] = eks[-1]*(self.ops['bn'])  # energy of boson w/o offset
        self.Ws['WL'] = WL.copy()
        self.sites += ['WL']

             # vs Bulla:  delta/2 = t0, epsilon/2 = e0, omega_i = eks[i], lambda_i/2 = vks[i]

    def getLanczos(self):

        time1 = time.time()
        db = self.ds[-1]

        a = [0]*(self.L-1)  # system = 1 site
        b = [0]*(self.L-1)

        # treat MPO as 2D matrix (just showing couplings)
        eks = np.array(self.eks)
        vks = np.array(self.vks)

        ## obtain vnorm = eta_0
        vnorm = np.sqrt( np.sum(vks**2) )
        # vnorm = np.sqrt( np.trapz( vks**2 , eks ) )

        print 'vnorm', vnorm
        v0 = vks/vnorm

        a0 = np.sum(eks*v0**2)   # v0*Hv0
        # a0 = np.trapz(eks*v0**2, eks)
        v1 = (eks - a0)*v0
        b0 = np.sqrt( np.dot(v1,v1) ) # 1./vnorm * np.sqrt( np.sum( (eks - a0)**2 * vks**2 ) )
        # b0 = np.sqrt( np.trapz(np.abs(v1)**2, eks) )
        print 'v1 norm', b0

        a[0] = a0
        b[0] = vnorm #/2./np.sqrt(np.pi)

        if b0 > 1.0e-8:
            v1 = v1/b0

            for i in range(1,self.L-1):  # just looping over bath sites
     
                a1 = np.sum(eks*v1**2)
                # a1 = np.trapz( eks*v1**2, eks )
                v2 = ( (eks-a1)*v1 - b0*v0 )
                b1 = np.sqrt( np.dot(v2, v2) )
                # b1 = np.sqrt( np.trapz( np.abs(v2)**2, eks ) )

                b[i] = b0
                a[i] = a1

                b0 = b1
                v0 = v1
                if b1 < 1.0e-8:
                    print 'small coupling so terminate', b1
                    print 'involves ', i+1, ' out of ', self.L-1, ' bath vectors'
                    continue
                else:
                    v1 = v2/b1
        else:    pass   # remaining couplings are 0
        
        try:            print 'remaining b1', b1
        except(UnboundLocalError):  print 'no coupling'   # b1 not defined

        print 'lanzcos', a
        print 'lanzcos', b

        # spin site mpo
        Wspin = self.Ws['WS']
        W  = np.zeros([1,2,2,4])
        W[0,:,:,0] = Wspin[0,:,:,0]
        W[0,:,:,1] = Wspin[0,:,:,1]
        W[0,:,:,2] = Wspin[0,:,:,1]
        W[0,:,:,3] = np.eye(2)
        mpo_sites = [ W.copy() ]
        
        # rotated bath sites
        db = self.ds[-1]
        i = 0
        while i < self.L-2:
            W  = np.zeros([4,db,db,4])
            W[0,:,:,0] = np.eye(db)
            W[1,:,:,0] = self.ops['b-']*np.conj(b[i])   # b[0] = vnorm, coupling to spin site
            W[2,:,:,0] = self.ops['b+']*b[i]
            W[3,:,:,0] = self.ops['bn']*a[i]
            W[3,:,:,1] = self.ops['b+']
            W[3,:,:,2] = self.ops['b-']
            W[3,:,:,3] = np.eye(db)
            mpo_sites.append(W.copy())
            i += 1

        # last bath site
        print i, len(a)
        W = np.zeros([4,db,db,1])
        W[0,:,:,0] = np.eye(db)
        W[1,:,:,0] = self.ops['b-']*np.conj(b[i])
        W[2,:,:,0] = self.ops['b+']*b[i]
        W[3,:,:,0] = self.ops['bn']*a[i]
        mpo_sites.append(W.copy())

        print 'lanczos tridiagonal vals'
        # print a, eks
        # print b, vks

        ###### check eigvals ######
        H1 = np.diag(np.append(self.e0,eks))
        H1[0,1:] = vks[:]# /2./np.sqrt(np.pi)
        H1[1:,0] = vks[:]# /2./np.sqrt(np.pi)

        # H2 = np.diag([1]+a)+np.diag([b[0]/2/np.sqrt(np.pi)] + b[1:],k=-1)+np.diag([b[0]/2/np.sqrt(np.pi)] + b[1:],k=1)
        H2 = np.diag([self.e0]+a)+np.diag([b[0]] + b[1:],k=-1)+np.diag([b[0]] + b[1:],k=1)

        print 'evals',np.sort(np.linalg.eigvals(H1))
        print 'evals',np.sort(np.linalg.eigvals(H2))
        


        # ### check full eigvals ###
        # sqdim = np.prod(self.ds)
        # df, db = 2, self.ds[-1]
        # H1full = np.einsum('ab,cd->acbd',self.ops['SX']*e0,np.eye(db*n_boson)).reshape(sqdim,sqdim)
        # bns = np.outer(eks,np.range(db)).reshape(-1)
        # H1full += np.einsum('ab,cd->acbd',np.eye(df),bns).reshape(sqdim,sqdim)
        # for x in range(n_boson):
        #     sq = df
        #     for xx in range(x):
        #         sq = sq*db
        #         boson_part = np.einsum('ab,cd->acbd',self.ops['SZ'],np.eye(db)).reshape(sq,sq)
        #     sq = sq*db
        #     boson_part = np.einsum('ab,cd->acbd',boson_part,(self.ops['b+']+self.ops['b-'])*vks[x]/2./np.sqrt(np.pi)).reshape(sq,sq)
        #     for xx in range(x,n_boson):
        #         sq = sq*db
        #         boson_part = np.einsum('ab,cd->acbd',boson_part,np.eye(db)).reshape(sqdim,sqdim)
        #     H1full += boson_part

        # print 'H1 full', np.linalg.eigvals(H1full)

        # H2full = np.einsum('ab,cd->acbd',self.ops['SX']*e0,np.eye(db*n_boson)).reshape(sqdim,sqdim)
        # V = np.einsum('ab,cd->acbd',self.ops['SZ']*e0,(self.ops['b+']+self.ops['b-'])*b[0]/2/np.sqrt(np.pi)).reshape(df*db,df*db)
        # H2full += 
        # for x in range(n_boson):
        #     sq = df
        #  
        # exit()

 
        return MPX.MPX( mpo_sites )


    def getHouseholder(self):
        ## see https://math.byu.edu/~schow/resources/householder.pdf (not a great reference)

        A = np.diag([self.e0]+list(self.eks)) 
        A[1:,0] = np.conj(self.vks)
        A[0,1:] = self.vks

        A_orig = A.copy()
        H_trans = np.eye(len(A))


        for i in range(self.L-1):

            x = A[i+1:,i].copy()
            xnorm = np.linalg.norm(x)
            alpha = np.sign(x[0])*xnorm

            r = np.sqrt(0.5*xnorm*(xnorm + np.abs(x[0])))
            u = x
            u[0] = u[0] + alpha
            u = u/2/r
            u = np.append( [0]*(i+1), u )

            P = np.eye(self.L) - 2*np.outer(u,u)
            A = np.einsum('ij,jk,kl->il',P,A,P)
            H_trans = np.matmul(H_trans,P)

        new_eks = np.diag(A)
        new_vks = np.diag(A,k=1)
        print 'householder', new_eks
        print 'householder', new_vks

        ex_A = np.diag(new_eks) + np.diag(new_vks,k=1) + np.diag(np.conj(new_vks),k=-1)
        assert np.all(np.abs(ex_A - A) < 1.0e-12), 'inaccurate householder transformation'

        # ## check evals
        evals1 = np.sort(np.linalg.eigvals(A_orig))
        evals2 = np.sort(np.linalg.eigvals(A))
        print 'evals', evals1
        print 'evals', evals2
        # assert np.all( np.abs(evals1-evals2) < 1.0e-12 ), 'evals error after householder transformation'

        assert(new_eks[0] == self.e0), 'householder changed site energy but should not have'
        a = new_eks[1:]
        b = new_vks[:]
        b = np.append( new_vks[0]*-1, new_vks[1:] )  # to match lanczos coeffs.  interestingly, doesn't make a difference?

        # spin site mpo
        Wspin = self.Ws['WS']
        W  = np.zeros([1,2,2,4])
        W[0,:,:,0] = Wspin[0,:,:,0]
        W[0,:,:,1] = Wspin[0,:,:,1]
        W[0,:,:,2] = Wspin[0,:,:,1]
        W[0,:,:,3] = np.eye(2)
        mpo_sites = [ W.copy() ]
        
        # rotated bath sites
        db = self.ds[-1]
        i = 0
        while i < self.L-2:
            W  = np.zeros([4,db,db,4])
            W[0,:,:,0] = np.eye(db)
            W[1,:,:,0] = self.ops['b-']*np.conj(b[i])   # b[0] = vnorm, coupling to spin site
            W[2,:,:,0] = self.ops['b+']*b[i]
            W[3,:,:,0] = self.ops['bn']*a[i]
            W[3,:,:,1] = self.ops['b+']
            W[3,:,:,2] = self.ops['b-']
            W[3,:,:,3] = np.eye(db)
            mpo_sites.append(W.copy())
            i += 1

        # last bath site
        print i, len(a)
        W = np.zeros([4,db,db,1])
        W[0,:,:,0] = np.eye(db)
        W[1,:,:,0] = self.ops['b-']*np.conj(b[i])
        W[2,:,:,0] = self.ops['b+']*b[i]
        W[3,:,:,0] = self.ops['bn']*a[i]
        mpo_sites.append(W.copy())

        return MPX.MPX( mpo_sites ), H_trans
     


class MPO_anderson(MPO):

    ## spin occupation quantum number:  u(p) or d(own), parity
        
    ## Anderson model:
    ## H =  
    ##   impurity:       \sum_q       eq     q+q   + U  u+ud+d
    ##   environment:  + \sum_rj,r'j' trj,rj rj+rj + cc
    ##   coupling:     + \sum_sq,rj   Vsq,rj sq+rj + cc
    ## (q,r = spin occupation #s, j = lattice site position)

    def __init__(self,n_bath):
        
        ## ordered like (up, down):  00, 01, 10, 11

        self.ops = {}
        self.ops['u+'] = np.array([[0,0,0,0],[0,0,0,0],[1,0,0,0],[0,1,0,0]])
        self.ops['d+'] = np.array([[0,0,0,0],[1,0,0,0],[0,0,0,0],[0,0,1,0]])
        self.ops['u-'] = np.array([[0,0,1,0],[0,0,0,1],[0,0,0,0],[0,0,0,0]])
        self.ops['d-'] = np.array([[0,1,0,0],[0,0,0,0],[0,0,0,1],[0,0,0,0]])
        self.ops['nu'] = np.array([[0,0,0,0],[0,0,0,0],[0,0,1,0],[0,0,0,1]])
        self.ops['nd'] = np.array([[0,0,0,0],[0,1,0,0],[0,0,0,0],[0,0,0,1]])
        self.ops['nn'] = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,1]])

        ## how to include sign change symmetries in MPO?


    
class MPO_noise(MPO):

   def __init__(self,n,bath):
       
      self.ops = paulis



####  defintion as sum of two-site operators ####
class TrotterOperator:
    def __init__(self,L,ns):
        
        self.L       = L
        self.nsites  = ns
        self.edges   = []    # which sites are interacting
        self.weights = []    # interaction strength
        self.ops     = None  # MPO operating on ns sites 

    
class Heisenberg_sum(TrotterOperator):   
    '''
    sum_i h_i S_i + J_i (S_i x S_(i+1))
    '''
    def __init__(self,L,hs=None,Js=None):
        
        self.L     = L
        self.ns    = 2
        self.edges = [(n,n+1) for n in range(L-1)]
        self.weights = 1.0

        if hs == None and Js == None:   # default AFM heisenberg 
            op_1 = np.zeros((1,2,2,3))
            op_2 = np.zeros((3,2,2,1))
            op_1[0,:,:,0] = sp.copy()
            op_1[0,:,:,1] = sm.copy()
            op_1[0,:,:,2] = sz.copy()
            op_2[0,:,:,0] = sm.copy()
            op_2[1,:,:,0] = sp.copy()
            op_2[2,:,:,0] = sz.copy()

            self.ops   = MPX.MPX( [op_1,  op_2],  phys_bonds=[(2,2),(2,2)] )
        else:
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
            
            if len(Js) <= 3:
                Js1 = np.where(Js != 0)[0]
                Js2 = []
                Dw = len(Js1) + 1
            else:
                Js1 = np.where(np.array(Js[:3]) != 0)[0]
                Js2 = np.where(np.array(Js[3:]) != 0)[0]
                Dw = len(Js1) + len(Js2)*2 + 1
 
            op_1 = np.zeros((1,2,2,Dw),dtype=np.complex128)
            op_2 = np.zeros((Dw,2,2,1),dtype=np.complex128)
            op_1L = np.zeros((1,2,2,Dw+1),dtype=np.complex128)
            op_2L = np.zeros((Dw+1,2,2,1),dtype=np.complex128)
            
            # identity terms
            op_2[0,:,:, 0] = np.eye(2) 
            
            # on site term
            for ind in range(3):
                if not hs[ind] == 0:
                    op_1[0,:,:,0]  = op_1[0,:,:,0].copy()  + hs[ind]*ops[singList[ind]].copy()
        
            for ii in range(len(Js1)):
                ind = Js1[ii]
                op_1[0,:,:,1+ii] = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+ii,:,:,0] = ops[coupList[ind][0]].copy()
                
            for ii in range(len(Js2)):
                ind = Js2[ii]+3
                op_2[1+len(Js1)+2*ii,:,:,0]   = ops[coupList[ind][0]].copy()
                op_1[0,:,:,1+len(Js1)+2*ii]  = ops[coupList[ind][1]].copy()*Js[ind]
                op_2[1+len(Js1)+2*ii+1,:,:,0] = ops[coupList[ind][1]].copy()
                op_1[-1,:,:,1+len(Js1)+2*ii]  = ops[coupList[ind][0]].copy()*Js[ind]

            op_1L[:,:,:,:-1] = op_1.copy()
            op_1L[0,:,:,-1]  = np.eye(2) 
            op_2L[:-1,:,:,:] = op_2.copy()
            op_2L[-1,:,:,0]  = op_1[0,:,:,0].copy()   # on site term

            self.opsL  = MPX.MPX( [op_1L, op_2L], phys_bonds=[(2,2),(2,2)] )
            self.ops   = MPX.MPX( [op_1,  op_2],  phys_bonds=[(2,2),(2,2)] )


    def getEdges(self, i):
        ''' 
        returns on-site term?, 
        returns list of all edges connected to site i 
        '''
        return [(i-1,i), (i,i+1)], [1.0, 1.0]



