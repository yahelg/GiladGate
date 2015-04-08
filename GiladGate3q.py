from __future__ import division

from qutip import *
from scipy import *
from scipy.signal import argrelextrema
from scipy.optimize import curve_fit

def operate(a,b):
    return vector_to_operator(a*operator_to_vector(b))
def unify(st):
    return st/st.norm()

def expectL3(a):
    ll = expect(ketbra(L3),a.rho)
    plot(a.timelist,real(ll))
    
def ketbra(ket,bra=None):
    if not bra:
        bra = ket
    return ket * bra.dag()

def func(x, a, c, d):
    return a*np.exp(-c*x)+d

def comp(x):
    if x.imag == 0:
        return '({0.real:.2f})'.format(x)
    elif x.real == 0:
        return '({0.imag:.2f}i)'.format(x)
    else:
        return '({0.real:.2f} + {0.imag:.2f}i)'.format(x)

def strket(a):
    return '+'.join([comp(a[i][0][0])+'|{0:03b}>'.format(i) for i in range(8) if a[i]!=0])

def strket(a):
    kett = [unify(ket('110') + ket('000')),unify(ket('110') - ket('000')),ket('100'),ket('010')]
    kett += [unify(ket('111') + ket('001')),unify(ket('111') - ket('001')),ket('101'),ket('011')]
    states = ['|L1>','|L3>','|E1>','|E2>','|O+>','|L2>','|e1>','|e2>']
    a = a.transform(kett)
    return '+'.join([comp(a[i][0][0])+states[i] for i in range(8) if a[i]!=0])
    
def printstates(b):
    eige = b.eigenstates()[0]
    eigstates = b.eigenstates()[1]
    states = [strket(eigstates[i].tidyup(0.001)) for i in range(len(eige)) if norm(eige[i])>0.001 ]
    ens = ['({0.real:.2f})'.format(eige[i]) for i in range(len(eige)) if norm(eige[i])>0.001]
    print '\n'.join(['%s : %s'%(i,j) for (i,j) in zip(ens,states)])

def showdecay(a):
    vals = real(expect(ketbra(L1),a.rho))
    smoothed = np.convolve(vals, np.ones(10)/10)
    maxi = argrelextrema(smoothed, np.greater)[0]
    d = a.data
    p0=(0.2,0.5*d.gamma*(d.g/d.delta)**2, 0.5)
    popt, pcov = curve_fit(func, a.timelist[maxi],vals[maxi],p0)
    plot(a.timelist,func(a.timelist,*popt))
    print 'decayrate: ',popt[1],' vs gamma*(g/delta)^2: ',d.gamma*(d.g/d.delta)**2
    return popt[1]

def showperiod(a,points=10,maxed=5):
    vals = real(expect(ketbra(L1),a.rho))
    smoothed = np.convolve(vals, np.ones(points)/points)
    maxi = argrelextrema(smoothed, np.greater)[0]
    pr = 1/(a.timelist[maxi[3]] - a.timelist[maxi[2]])
    d = a.data
    print 'period: ',pr,' vs g^2/(4*delta): ',d.g**2/(4*d.delta)
    plot(a.timelist[:maxi[maxed]],vals[:maxi[maxed]])
    plot(a.timelist[maxi[[2,3]]],vals[maxi[[2,3]]],'r*')
    return pr
    
L1 = unify(ket('110')+ket('000'))
L2 = unify(ket('111')-ket('001'))
L3 = unify(ket('110')-ket('000'))
I = tensor(qeye(2),qeye(2),qeye(2))

#Error Correction
err = [tensor(sigmax(),qeye(2),qeye(2)),tensor(sigmay(),qeye(2),qeye(2))]
Pe = sum([ketbra(op*L1) + ketbra(op*L2) for op in err])
U = sum([ketbra(op*L1,L1) + ketbra(op*L2,L2) for op in err])
U = U + U.dag()
Pc = I - Pe

class params(object):
    def __init__(self):
        # Parameters
        self.g = 0.1
        self.gamma = 0.1
        self.gec = 1
        self.delta = 10
    def rho0(self):
        return ketbra(L1)
    def H0(self):
        return -self.delta*(ketbra(L1)+ketbra(L2))        
    def Ht(self):
        #Hamiltonian and changes
        #Might cause wrong error correction because L1 and L2 rotate
        #H0 = self.delta*ketbra(L3)
        V = self.g/2 * (ketbra(L1,L3) + ketbra(L2,L3))
        V = V + V.dag()
        return self.H0() + V
    def Ld(self):
        ld = lindblad_dissipator(tensor(sigmam(),qeye(2),qeye(2)))
        # Add upwords dissipation
        ld += lindblad_dissipator(tensor(sigmap(),qeye(2),qeye(2)))
        return self.gamma * ld
    def Lec(self):
        lec = sprepost(Pc,Pc)+ sprepost(U*Pe,Pe*U) -spre(I)
        return  self.gec * lec
    def correct(self):
        return sprepost(Pc,Pc)+ sprepost(U*Pe,Pe*U)
    def printParams(self):
        print 'g: %f\ngamma: %f\ngec: %f\ndelta: %f'%(self.g,self.gamma,self.gec,self.delta)

A1 = ketbra(L1,L1) + ketbra(L1,L3) + ketbra(L2,L2)
A2 = ketbra(L1,L1) + ketbra(-L1,L3) + ketbra(L2,L2)
class paramsDecayAndCorrection(params):
    def Ld(self):
        return self.gamma/2 * (sprepost(A1,A1.dag()) + sprepost(A2,A2.dag()) - 2*spre(I))
    def Lec(self):
        pass

class paramsSigma(params):
    def Ht(self):
        V = self.g/2 * tensor(sigmaz(),qeye(2),qeye(2))
        V += self.g/2 * tensor(qeye(2),qeye(2),sigmax())
        return self.H0() + V

class rhoResult(object):
    def __init__(self,data,rho,timelist=None):
        if timelist==None:
            timelist = xrange(len(rho))
        self.data = data
        self.rho = rho
        self.timelist = timelist
    def oo(self):
        o1 = expect(ketbra(L1),self.rho)
        o2 = expect(ketbra(L2),self.rho)
        return [o1,o2]
    def pop(self):
        [o1,o2] = self.oo()
        plot(self.timelist, real(o1), 'b',self.timelist, real(o2), 'r')
        xlabel('Time')
        ylabel('Occupation probability')
        title('Occupation of L1 and L2 vs time')        
    def decay(self):
        eigs = [real(b.eigenenergies()[-1]) for b in self.rho]
        xlabel('Time')
        ylabel('biggest eigenvalue')
        title('decay rate')
        plot(self.timelist,eigs)
    def decayrate(self,points=10,p0=(1,0.001, 0.5)):
        # for local maxima
        vals = expect(ketbra(L1),self.rho)
        maxi = np.append([0],argrelextrema(vals, np.greater)[0])
        
        t = array([float(real(i)) for i in self.timelist[maxi]])
        v = array([float(real(i)) for i in vals[maxi]])
        
        popt, pcov = curve_fit(func, t[:points], v[:points], p0)
        return popt

def exper(data=params(),eflag=True,cflag=True,time=None,steps=5000):
    c_op_list = []
    if eflag:
        c_op_list.append(data.Ld())
    if cflag:
        c_op_list.append(data.Lec())
    if not time:
        time = 10*2*pi/(data.g**2/data.delta)
    timelist = linspace(0,time,steps)
    out = mesolve(data.Ht(),data.rho0(),timelist,c_op_list,[])
    return rhoResult(data,out.states,timelist)
