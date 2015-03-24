#many definitions
from qutip import *
from pylab import *
from qutip.tensor import tensor
from qutip.states import basis

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

p0 = tensor(psiplus,basis(2,0),basis(2,0))
p1 = tensor(psiminus,basis(2,1),basis(2,1))
pp0 = tensor(psiplus,basis(2,0))
pm1 = tensor(psiminus,basis(2,1))

# Parameters
w = 10
delta = 1
g = 0.1
Om = 0.1
wm = 10
wp = 1
dr = Om*g/delta/4
gamma = 0.1

#Hamiltonian and changes
def addqubit(op):
    return tensor(op,qeye(2))
pp = addqubit(pp)
mm = addqubit(mm)
zz = addqubit(zz)
onon = addqubit(onon)
dd = addqubit(down*down.dag())

H0 = wp*pp + wm*mm - wp*zz -wm*onon + 100*dd
Hd1 = g * tensor(sz,qeye(2),qeye(2),sx)
Hd2 = Om * tensor(qeye(2),qeye(2),sx,qeye(2))
args = {'w': w,'delta':delta,'wm':wm,'wp':wp}
Ht = [H0, [Hd1, 'sin((wm-wp-delta)*t)'],[Hd2, 'sin((wm-wp-delta)*t)']]

def sum3(a):
    return a[0] + a[1] + a[2]

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

#Error Correction
ey = tensor(qeye(2),qeye(2))
err = [tensor(sx,ey),tensor(sy,ey),tensor(sz,ey)]
P = [(op*pp0)*(op*pp0).dag() + (op*pm1)*(op*pm1).dag() for op in err]
U = [(op*pp0)*pp0.dag() + (op*pm1)*pm1.dag() for op in err]
U = [a + a.dag() for a in U]
P = [addqubit(p) for p in P]
U = [addqubit(u) for u in U]
el = tensor(qeye(2),qeye(2),qeye(2),qeye(2)) - sum3(P)
target = (1/sqrt(2)*(p0 -1j*p1))

class rhoResult:
    def __init__(self,rho):
        self.w = w
        self.delta = delta
        self.g = g
        self.Om = Om
        self.wm = wm
        self.wp = wp
        self.dr = dr
        self.gamma = gamma
        self.rho = rho
    def oo(self):
        o1 = expect(p1 * p1.dag(),self.rho)
        o2 = expect(p0 * p0.dag(),self.rho)
        return [o1,o2]
    def showr(self):
        [o1,o2] = self.oo()
        tlist = linspace(0,1,len(o1))
        plot(tlist, real(o1), 'b',tlist, real(o2), 'b--')
        xlabel('Time')
        ylabel('Occupation probability')
        title('Occupation of L1 and L2 vs time')
        show()    
    def showtarg(self,targ=target):
        tlist = linspace(0,1,len(self.rho))
        plot(tlist,real(expect(targ*targ.dag(),self.rho)))
        xlabel('Time')
        ylabel('Target occupation')
        title('Target occupation vs time')
        show()
        
I = tensor(qeye(2),qeye(2),qeye(2),qeye(2))
kraus = [u*p for (u,p) in zip(U,P)]
kraus.append(I-(P[0] + P[1] + P[2]))
kraus.append(1j*I)
kraus = [sqrt(2*gamma)*kr for kr in kraus]
up = [u*p for (u,p) in zip(U,P)]
pu = [p*u for (u,p) in zip(U,P)]
rest = I-(P[0] + P[1] + P[2])
corr = 100*gamma*(sprepost(up[0],pu[0])+sprepost(up[1],pu[1])+sprepost(up[2],pu[2])+sprepost(rest,rest)+spre(-I))
gc = 100*gamma
smt = tensor(sm,qeye(2),qeye(2),qeye(2))

def correct(rho):
    a = sum3([u*p*rho*p*u for (p,u) in zip(P,U)]) + rest*rho*rest
    return a

def experp(eflag=True,cflag=True,time=pi/dr):
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    c_op_list = []
    if eflag:
        #c_op_list.append(lindblad_dissipator(sqrt(gamma)*tensor(sm,qeye(2),qeye(2),qeye(2))))
        c_op_list.append(lindblad_dissipator(sqrt(gamma)*((smt*p0)*p0.dag()+(smt*p1)*p1.dag())))
    if cflag:
        pass
        #c_op_list.append(kraus_to_super(kraus))
        #c_op_list += [gc*sprepost(up[i],pu[i]) for i in range(3)]
        #c_op_list.append(gc*sprepost(rest,rest)+gc*spre(-I))
        #c_op_list.append(corr)
    output = []
    timelist = linspace(0,time,1000)
    rho0 = p0 * p0.dag()
    for tlist in chunks(timelist, 10):
        out = mesolve(Ht,rho0,tlist,c_op_list,[],args)
        output += out.states
        if cflag:
            rho0 = correct(output[-1])
        else:
            rho0 = output[-1]
    res = rhoResult(output)
    res.eflag = eflag
    res.cflag = cflag
    return res

def exper(eflag=True,cflag=True,time=pi/dr):
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    c_op_list = []
    if eflag:
        #c_op_list.append(lindblad_dissipator(sqrt(gamma)*tensor(sm,qeye(2),qeye(2),qeye(2))))
        #c_op_list.append(lindblad_dissipator(sqrt(gamma)*((smt*p0)*p0.dag()+(smt*p1)*p1.dag())))
        ot = I - (p0*p0.dag()+p1*p1.dag())
        c_op_list.append(lindblad_dissipator(sqrt(gamma)*((smt*p0)*p0.dag()+(smt*p1)*p1.dag()+ot)))
    if cflag:
        #c_op_list.append(kraus_to_super(kraus))
        c_op_list += [gc*sprepost(up[i],pu[i]) for i in range(3)]
        c_op_list.append(gc*sprepost(rest,rest)+gc*spre(-I))
        #c_op_list.append(corr)
    output = []
    tlist = linspace(0,time,1000)
    rho0 = p0 * p0.dag()
    out = mesolve(Ht,rho0,tlist,c_op_list,[],args)
    res = rhoResult(out.states)
    res.eflag = eflag
    res.cflag = cflag
    return res