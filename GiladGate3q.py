#many definitions
from qutip import *
from pylab import *
from qutip.tensor import tensor
from qutip.states import basis
import copy

sx = sigmax()
sy = sigmay()
sz = sigmaz()
sm = destroy(2)

sz1 = tensor(sz,qeye(2),qeye(2))
sz2 = tensor(qeye(2),sz,qeye(2))
sz3 = tensor(qeye(2),qeye(2),sz)

psiplus = 1/sqrt(2)*(tensor(basis(2,1),basis(2,1))+tensor(basis(2,0),basis(2,0)));
pp = tensor(psiplus * psiplus.dag(),qeye(2))
psiminus = 1/sqrt(2)*(tensor(basis(2,1),basis(2,1))-tensor(basis(2,0),basis(2,0)));
mm = tensor(psiminus * psiminus.dag(),qeye(2))
zz = tensor(qeye(2),qeye(2),basis(2,0)*basis(2,0).dag())
onon = tensor(qeye(2),qeye(2),basis(2,1)*basis(2,1).dag())
up = tensor(psiminus,basis(2,0))
down = tensor(psiplus,basis(2,1))
dd = down*down.dag()

p0 = tensor(psiplus,basis(2,0))
p1 = tensor(psiminus,basis(2,1))
pp0 = tensor(psiplus,basis(2,0))
pm1 = tensor(psiminus,basis(2,1))

#Error Correction
ey = tensor(qeye(2),qeye(2))
err = [tensor(sx,ey),tensor(sy,ey)]
P = [(op*pp0)*(op*pp0).dag() + (op*pm1)*(op*pm1).dag() for op in err]
U = [(op*pp0)*pp0.dag() + (op*pm1)*pm1.dag() for op in err]
U = [a + a.dag() for a in U]
el = tensor(qeye(2),qeye(2),qeye(2)) - sum(P)
target = (1/sqrt(2)*(p0 -1j*p1))

I = tensor(qeye(2),qeye(2),qeye(2))
#kraus = [u*p for (u,p) in zip(U,P)]
#kraus.append(I-(P[0] + P[1] + P[2]))
#kraus.append(1j*I)
#kraus = [sqrt(2*gamma)*kr for kr in kraus]
up = [u*p for (u,p) in zip(U,P)]
pu = [p*u for (u,p) in zip(U,P)]
rest = I-sum(P)
#corr = 100*gamma*(sprepost(up[0],pu[0])+sprepost(up[1],pu[1])+sprepost(up[2],pu[2])+sprepost(rest,rest)+spre(-I))
smt = tensor(sm,qeye(2),qeye(2))
spt = tensor(sm.dag(),qeye(2),qeye(2))

class params:
    def __init__(self):
        # Parameters
        self.w = 10
        self.delta = 1
        self.g = 0.1
        self.Om = 0.1
        self.wm = 10
        self.wp = 1
        self.dr = self.Om*self.g/self.delta/4
        self.gamma = 0.1
        self.gc = 100*self.gamma
    def Ht(self):
        #Hamiltonian and changes
        H0 = self.wp*pp + self.wm*mm - self.wp*zz -self.wm*onon + 100*dd
        Hd1 = self.g * tensor(sz,qeye(2),qeye(2))
        Hd2 = self.Om * tensor(qeye(2),qeye(2),sx)
        Ht1 = [H0, [Hd1, 'sin((wm-wp-delta)*t)'],[Hd2, 'sin((wm-wp-delta)*t)']]
        return Ht1
    def args(self):
        args1 = {'w': self.w,'delta':self.delta,'wm':self.wm,'wp':self.wp}
        return args1
    def corr(self):
        corr1 = self.gc * sum([sprepost(up[i],pu[i]) for i in range(2)])
        corr1 += self.gc * sum([sprepost(P[i],P[i]) for i in range(2)])
        corr1 += self.gc * sum([spre(-P[i]) for i in range(2)])
        corr1 += self.gc * sum([spost(-P[i]) for i in range(2)])
        return corr1

data = params()        
        
def sum3(a):
    return a[0] + a[1] + a[2]

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

class rhoResult:
    def __init__(self,rho,timelist=None):
        if timelist==None:
            timelist = xrange(len(rho))
        self.data = copy.copy(data)
        self.rho = rho
        self.timelist = timelist
    def oo(self):
        o1 = expect(p1 * p1.dag(),self.rho)
        o2 = expect(p0 * p0.dag(),self.rho)
        return [o1,o2]
    def showr(self):
        [o1,o2] = self.oo()
        #tlist = linspace(0,1,len(o1))
        plot(self.timelist, real(o1), 'b',self.timelist, real(o2), 'b--')
        xlabel('Time')
        ylabel('Occupation probability')
        title('Occupation of L1 and L2 vs time')
        show()    
    def showtarg(self,targ=target):
        #tlist = linspace(0,1,len(self.rho))
        plot(self.timelist,real(expect(targ*targ.dag(),self.rho)))
        xlabel('Time')
        ylabel('Target occupation')
        title('Target occupation vs time')
        show()
    def decay(self):
        eigs = [real(b.eigenenergies()[-1]) for b in self.rho]
        xlabel('Time')
        ylabel('biggest eigenvalue')
        title('decay rate')
        plot(self.timelist,eigs)
        


def correct(rho):
    a = sum([u*p*rho*p*u for (p,u) in zip(P,U)]) + rest*rho*rest
    return a

def experp(eflag=True,cflag=True,time=pi/data.dr,steps=1000,chnk=10):  
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    c_op_list = []
    gamma = data.gamma
    if eflag:
        #c_op_list.append(lindblad_dissipator(sqrt(gamma)*smt))
        c_op_list.append(lindblad_dissipator(sqrt(gamma)*((smt*p0)*p0.dag()+(smt*p1)*p1.dag())))
    if cflag:
        pass
        #c_op_list.append(kraus_to_super(kraus))
        #c_op_list += [gc*sprepost(up[i],pu[i]) for i in range(3)]
        #c_op_list.append(gc*sprepost(rest,rest)+gc*spre(-I))
        #c_op_list.append(corr)
    output = []
    timelist = linspace(0,time,steps)
    rho0 = p0 * p0.dag()
    for tlist in chunks(timelist, chnk):
        out = mesolve(data.Ht(),rho0,tlist,c_op_list,[],data.args())
        output += out.states
        if cflag:
            rho0 = correct(output[-1])
        else:
            rho0 = output[-1]
    res = rhoResult(output,timelist)
    res.eflag = eflag
    res.cflag = cflag
    return res

def exper(eflag=True,cflag=True,time=pi/data.dr,steps=1000):
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    gamma = data.gamma
    c_op_list = []
    if eflag:
        c_op_list.append(lindblad_dissipator(sqrt(gamma)*smt))
        c_op_list.append(lindblad_dissipator(sqrt(gamma)*spt))
#        c_op_list.append(lindblad_dissipator(sqrt(gamma)*((smt*p0)*p0.dag()+(smt*p1)*p1.dag())))
#        c_op_list.append(lindblad_dissipator(sqrt(gamma)*((spt*p0)*p0.dag()+(spt*p1)*p1.dag())))
        #ot = I - (p0*p0.dag()+p1*p1.dag())
        #c_op_list.append(lindblad_dissipator(sqrt(gamma)*((smt*p0)*p0.dag()+(smt*p1)*p1.dag()+ot)))
    if cflag:
        #c_op_list.append(kraus_to_super(kraus))
        #c_op_list += [gc*sprepost(up[i],pu[i]) for i in range(3)]
        #c_op_list.append(gc*sprepost(rest,rest)+gc*spre(-I))
        c_op_list.append(data.corr())
    timelist = linspace(0,time,steps)
    rho0 = p0 * p0.dag()
    out = mesolve(data.Ht(),rho0,timelist,c_op_list,[],data.args())
    res = rhoResult(out.states,timelist)
    res.eflag = eflag
    res.cflag = cflag
    return res