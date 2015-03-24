
# coding: utf-8

# In[11]:




# In[1]:

from qutip import *
from pylab import *
from qutip.tensor import tensor
from qutip.states import basis

#many definitions

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

p0 = tensor(psiplus,basis(2,0))
p1 = tensor(psiminus,basis(2,1))
pp0 = tensor(psiplus,basis(2,0))
pm1 = tensor(psiminus,basis(2,1))

# For Error Correction
X = tensor(sx,qeye(2),qeye(2))
Y = tensor(sy,qeye(2),qeye(2))
Z = tensor(sz,qeye(2),qeye(2))
Px = (X*pp0)*(X*pp0).dag() + (X*pm1)*(X*pm1).dag() 
Py = (Y*pp0)*(Y*pp0).dag() + (Y*pm1)*(Y*pm1).dag() 
Pz = (Z*pp0)*(Z*pp0).dag() + (Z*pm1)*(Z*pm1).dag() 
Ux =  (X*pp0)*pp0.dag() + (X*pm1)*pm1.dag() 
Uy =  (Y*pp0)*pp0.dag() + (Y*pm1)*pm1.dag() 
Uz =  (Z*pp0)*pp0.dag() + (Z*pm1)*pm1.dag()
Ux = Ux + Ux.dag()
Uy = Uy + Uy.dag()
Uz = Uz + Uz.dag() 
el = tensor(qeye(2),qeye(2),qeye(2)) - Px - Py - Pz

# Parameters
w = 10
delta = 1
g = 0.1
Om = 0.1
wm = 10
wp = 1
dr = Om*g/delta/4
gamma = 0.05


# In[2]:

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
    def showtarg(self):
        target = (1/sqrt(2)*(p0 -1j*p1))
        plot(tlist,real(expect(target*target.dag(),self.rho)))
        xlabel('Time')
        ylabel('Target occupation')
        title('Target occupation vs time')
        show()


# In[ ]:

#Hamiltonian
H0 = wp*pp + wm*mm - wp*zz -wm*onon + 100*down*down.dag()
Hd1 = g * tensor(sz,qeye(2),qeye(2))
Hd2 = Om * tensor(qeye(2),qeye(2),sx)
args = {'w': w,'delta':delta,'wm':wm,'wp':wp}
Ht = [H0, [Hd1, 'sin((wm-wp-delta)*t)'],[Hd2, 'sin((wm-wp-delta)*t)']]


# In[11]:

# functions

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in xrange(0, len(l), n):
        yield l[i:i+n]

def tens(i,j,k):
    return tensor(basis(2,i),basis(2,j),basis(2,k))

def correct(rho):
    a = Ux*Px*rho*Px*Ux + Uy*Py*rho*Py*Uy + Uz*Pz*rho*Pz*Uz + el*rho*el
    
    return a
def exper(eflag=True,cflag=True,time=pi/dr):

    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    c_op_list = []
    gamma = 0.001
    if eflag:
        c_op_list.append(sqrt(gamma)*tensor(sm,qeye(2),qeye(2)))
       
    output = []
    timelist = linspace(0,time,1000)
    rho0 = p0 * p0.dag()
    for tlist in chunks(timelist, 100):
        out = mesolve(Ht,rho0,tlist,c_op_list,[],args)
        output += out.states
        if cflag:
            rho0 = correct(output[-1])
        else:
            rho0 = output[-1]
    return output

def oo(output):
    o1 = expect(p1 * p1.dag(),output)
    o2 = expect(p0 * p0.dag(),output)
    return [o1,o2]


# In[14]:

# Real work
[o1,o2] = oo(exper(False))
[o1c,o2c] = oo(exper())


# In[22]:

# Show results:
tlist = linspace(0,1,len(o1))
plot(tlist, real(o1), 'b',tlist, real(o2), 'b--',tlist,real(o1c),'r',tlist,real(o2c),'r--')
xlabel('Time')
ylabel('Occupation probability')
title('Excitation probabilty of qubit')
show()    


# In[25]:

def showr(o1,o2):
    tlist = linspace(0,1,len(o1))
    plot(tlist, real(o1), 'b',tlist, real(o2), 'b--')
    xlabel('Time')
    ylabel('Occupation probability')
    title('Excitation probabilty of qubit')
    show()    


# In[60]:

# Let's try to find the fidelity of the operation without errors.
out = exper(False,False,pi/dr/2)
[o1,o2] = oo(out)
showr(o1,o2)


# In[46]:

print len(out)
[o1,o2] = oo(out[:500])
showr(o1,o2)


# In[85]:

# Expectation value without errors
def showtarg(out):
    target = (1/sqrt(2)*(p0 -1j*p1))
    plot(tlist,real(expect(target*target.dag(),out)))
    show()

showtarg(out)
    #print (target.dag() * out[500] * target)


# In[86]:

# Let's try to find the fidelity of the operation with errors.
oute = exper(True,False,pi/dr/2)
showtarg(oute)


# In[88]:

# Let's try to find the fidelity of the operation with error correction.
outc = exper(True,True,pi/dr/2)
showtarg(outc)


## Hello

# In[116]:

target = (1/sqrt(2)*(p0 -1j*p1))
def ret(op):
    return real(expect(target*target.dag(),op))

plot(tlist,ret(out),tlist,ret(oute),tlist,ret(outc))
show()


### Now we'll move on to the extended gate:

# In[4]:

#many definitions

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

#Error Correction
ey = tensor(qeye(2),qeye(2))
err = [tensor(sx,ey),tensor(sy,ey),tensor(sz,ey)]
P = [(op*pp0)*(op*pp0).dag() + (op*pm1)*(op*pm1).dag() for op in err]
U = [(op*pp0)*pp0.dag() + (op*pm1)*pm1.dag() for op in err]
U = [a + a.dag() for a in U]
P = [addqubit(p) for p in P]
U = [addqubit(u) for u in U]
el = tensor(qeye(2),qeye(2),qeye(2),qeye(2)) - sum(P)
target = (1/sqrt(2)*(p0 -1j*p1))

def correct(rho):
    a = sum([u*p*rho*p*u for (p,u) in zip(P,U)]) + el*rho*el
    return a

def exper(eflag=True,cflag=True,time=pi/dr):
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    c_op_list = []
    if eflag:
        c_op_list.append(sqrt(gamma)*tensor(sm,qeye(2),qeye(2),qeye(2)))
       
    output = []
    timelist = linspace(0,time,10000)
    rho0 = p0 * p0.dag()
    for tlist in chunks(timelist, 10):
        out = mesolve(Ht,rho0,tlist,c_op_list,[],args)
        output += out.states
        if cflag:
            rho0 = correct(output[-1])
        else:
            rho0 = output[-1]
    return output

def oo(output):
    o1 = expect(p1 * p1.dag(),output)
    o2 = expect(p0 * p0.dag(),output)
    return [o1,o2]

def showtarg(out):
    plot(tlist,real(expect(target*target.dag(),out)))
    show()

def ret(op):
    return real(expect(target*target.dag(),op))


# In[127]:

# Let's try to find the fidelity of the operation without errors.
outg = exper(False,False,pi/dr/2)
[o1,o2] = oo(outg)
showr(o1,o2)


# In[128]:

[o1,o2] = oo(outg)
showr(o1,o2)


# In[150]:

# Let's try to find the fidelity of the operation with errors.
outge = exper(True,False,pi/dr/2)


# In[190]:

# Let's try to find the fidelity of the operation with error correction.
outgc = exper(True,True,pi/dr/2)
plot(tlist,ret(outg),tlist,ret(outge),tlist,ret(outgc))
show()


# In[168]:

a = outge[-1]
targ = (1/sqrt(2)*(p0 -p1))
targ = p1
expect(targ*targ.dag(),a)
a


# In[182]:

ps0 = tensor(1/sqrt(2)*(basis(2,0) + basis(2,1)),basis(2,0))
ps1 = tensor(1/sqrt(2)*(basis(2,0) - basis(2,1)),basis(2,1))
Hs = g * tensor(sz,sx)
targetsim = 1/sqrt(2) * (ps0 - 1j*ps1)

def simple(eflag=True,time=pi/dr):
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    c_op_list = []
    if eflag:
        c_op_list.append(sqrt(gamma)*tensor(sm,qeye(2)))
       
    output = []
    timelist = linspace(0,time,1000)
    rho0 = ps0 * ps0.dag()
    for tlist in chunks(timelist, 100):
        out = mesolve(Hs,rho0,tlist,c_op_list,[],args)
        output += out.states
        rho0 = output[-1]
    return output
def oos(output):
    o1 = expect(ps1 * ps1.dag(),output)
    o2 = expect(ps0 * ps0.dag(),output)
    return [o1,o2]
def retsim(op):
    return real(expect(targetsim*targetsim.dag(),op))


# In[173]:

# Let's try to find the fidelity of the operation without errors.
outgsimple = simple(False,pi/g/2)


# In[175]:

[o1,o2] = oos(outgsimple)
showr(o1,o2)


# In[174]:

outgesimple = simple(True,pi/g/2)
[o1,o2] = oos(outgesimple)
showr(o1,o2)


# In[183]:

plot(tlist,retsim(outgsimple),tlist,retsim(outgesimple))
show()


# In[94]:

I = tensor(qeye(2),qeye(2),qeye(2),qeye(2))
kraus = [u*p for (u,p) in zip(U,P)]
kraus.append(I-(P[0] + P[1] + P[2]))
kraus.append(1j*I)
kraus = [sqrt(2*gamma)*kr for kr in kraus]
up = [u*p for (u,p) in zip(U,P)]
pu = [p*u for (u,p) in zip(U,P)]
rest = I-(P[0] + P[1] + P[2])
corr = 100*gamma*(sprepost(up[0],pu[0])+sprepost(up[1],pu[1])+sprepost(up[2],pu[2])+sprepost(rest,rest)+spre(-I))
gc = 10*gamma
smt = tensor(sm,qeye(2),qeye(2),qeye(2))
def exper(eflag=True,cflag=True,time=pi/dr):
    #
    # evolve and system subject to the time-dependent hamiltonian
    #
    c_op_list = []
    if eflag:
        #c_op_list.append(lindblad_dissipator(sqrt(gamma)*tensor(sm,qeye(2),qeye(2),qeye(2))))
        c_op_list.append(lindblad_dissipator(sqrt(gamma)*((smt*p0)*p0.dag()+(smt*p1)*p1.dag())))
    if cflag:
        #c_op_list.append(kraus_to_super(kraus))
        c_op_list += [gc*sprepost(up[i],pu[i]) for i in range(3)]
        c_op_list.append(gc*sprepost(rest,rest)+gc*spre(-I))
        #c_op_list.append(corr)
    output = []
    tlist = linspace(0,time,10000)
    rho0 = p0 * p0.dag()
    out = mesolve(Ht,rho0,tlist,c_op_list,[],args)
    res = rhoResult(out.states)
    res.eflag = eflag
    res.cflag = cflag
    return res



# In[ ]:




# In[5]:

# Parameters
w = 10
delta = 1
g = 0.1
Om = 0.1
wm = 10
wp = 1
dr = Om*g/delta/4
gamma = 0.1


# In[49]:

# Let's try to find the fidelity of the operation with errors.
outg = exper(True,False,pi/dr/2)
[o1,o2] = oo(outg)
showr(o1,o2)


# In[29]:

def oo(output):
    o1 = expect(p1 * p1.dag(),output)
    o2 = expect(p0 * p0.dag(),output)
    return [o1,o2]

def showtarg(out):
    plot(tlist,real(expect(target*target.dag(),out)))
    show()

def ret(op):
    return real(expect(target*target.dag(),op))

[o1,o2] = oo(outg)
showr(o1,o2)


# In[66]:

# Let's try to find the fidelity of the operation with errors and correction.
oute = exper(True,True,pi/dr/2)
[o1,o2] = oo(oute)
showr(o1,o2)


# In[73]:

# Let's try to find the fidelity of the operation with errors and correction.
oute = exper(True,True,pi/dr/2)
[o1,o2] = oo(oute)
showr(o1,o2)


# In[68]:

# Let's try to find the fidelity of the operation with errors and correction.
out = exper(False,False,pi/dr/2)
[o1,o2] = oo(out)
showr(o1,o2)


# In[79]:

r = rhoResult(out)
r.showr()
w = 100
r.w


# In[8]:

out = []


# In[89]:

out.append(exper(True,False))
out[-1].showr()


# In[90]:

out.append(exper(True,True))
out[-1].showr()


### Interesting. The correction although being 10 times stronger couldn't correct the decay. Let's try 100

# In[92]:

out.append(exper(True,True))
out[-1].showr()


# In[95]:

out.append(exper(True,True))
out[-1].showr()


### Let's try with discrete correction.

# In[7]:

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


# In[8]:

out = []


# In[9]:

I = tensor(qeye(2),qeye(2),qeye(2),qeye(2))
kraus = [u*p for (u,p) in zip(U,P)]
kraus.append(I-(P[0] + P[1] + P[2]))
kraus.append(1j*I)
kraus = [sqrt(2*gamma)*kr for kr in kraus]
up = [u*p for (u,p) in zip(U,P)]
pu = [p*u for (u,p) in zip(U,P)]
rest = I-(P[0] + P[1] + P[2])
corr = 100*gamma*(sprepost(up[0],pu[0])+sprepost(up[1],pu[1])+sprepost(up[2],pu[2])+sprepost(rest,rest)+spre(-I))
gc = 10*gamma
smt = tensor(sm,qeye(2),qeye(2),qeye(2))

def correct(rho):
    a = sum3([u*p*rho*p*u for (p,u) in zip(P,U)]) + rest*rho*rest
    return a

def exper(eflag=True,cflag=True,time=pi/dr):
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
    timelist = linspace(0,time,10000)
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


# In[10]:

out.append(exper(True,True))
out[-1].showr()


# In[15]:



