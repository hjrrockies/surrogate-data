import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn
from scipy.special import kv, gamma

# 1-d Gaussian processes for functions with derivative data

class DKernel:
    def __init__(self,func,**kernel_args):
        self.func = func
        self.kernel_args = kernel_args

    def __call__(self,t,dt):
        return self.func(t,dt,**self.kernel_args)

    def __add__(self,other):
        def func(t,dt,kernel_args1,kernel_args2):
            return self.func(t,dt,**kernel_args1)+other.func(t,dt,**kernel_args2)
        kernel_args = {'kernel_args1':self.kernel_args,'kernel_args2':other.kernel_args}
        return DKernel(func,**kernel_args)

    def __sub__(self,other):
        def func(t,dt,kernel_args1,kernel_args2):
            return self.func(t,dt,**kernel_args1)-other.func(t,dt,**kernel_args2)
        kernel_args = {'kernel_args1':self.kernel_args,'kernel_args2':other.kernel_args}
        return DKernel(func,**kernel_args)

    def __mul__(self,other):
        def func(t,dt,kernel_args1,kernel_args2):
            return self.func(t,dt,**kernel_args1)*other.func(t,dt,**kernel_args2)
        kernel_args = {'kernel_args1':self.kernel_args,'kernel_args2':other.kernel_args}
        return DKernel(func,**kernel_args)

class DSquareExp(DKernel):
    @staticmethod
    def func(t,dt,lengthscale,tau):
        d11 = np.subtract.outer(t,t)
        d12 = np.subtract.outer(dt,t)
        d22 = np.subtract.outer(dt,dt)
        n,m = len(t),len(dt)
        a = np.empty((n+m,n+m))

        # state observations
        a[:n,:n] = np.exp(-0.5*(d11/lengthscale)**2)
        np.fill_diagonal(a[:n,:n],1+np.finfo('float64').eps**0.5)
        a[:n,:n] = tau**2*a[:n,:n]

        # cross terms
        a[n:,:n] = -(tau/lengthscale)**2*d12*np.exp(-0.5*(d12/lengthscale)**2)
        a[:n,n:] = a[n:,:n].T

        # derivative observations
        a[n:,n:] = (tau/lengthscale**2)**2*(lengthscale**2-d22**2)*np.exp(-0.5*(d22/lengthscale)**2)
        np.fill_diagonal(a[n:,n:],(tau/lengthscale)**2+np.finfo('float64').eps**0.5)
        return .5*(a+a.T)

    def __init__(self,lengthscale=1.0,tau=1.0):
        self.kernel_args = {'lengthscale':lengthscale,'tau':tau}

class DMatern(DKernel):
    @staticmethod
    def func(t,dt,nu,lengthscale,tau):
        d11 = np.subtract.outer(t,t)
        d12 = np.subtract.outer(dt,t)
        d22 = np.subtract.outer(dt,dt)
        d11[d11 == 0] += np.finfo(float).eps
        d12[d12 == 0] += np.finfo(float).eps
        d22[d22 == 0] += np.finfo(float).eps

        n,m = len(t),len(dt)
        a = np.empty((n+m,n+m))

        # if nu == 0.5:
        #     # state observations
        #     a[:n,:n] = tau**2*np.exp(-np.abs(d11)/lengthscale)
        #
        #     # cross terms
        #     a[n:,:n] = (-tau**2*d12/(lengthscale*np.abs(d12)))*np.exp(-np.abs(d12)/lengthscale)
        #     a[:n,n:] = a[n:,:n].T
        #
        #     # derivative observations
        #     a[n:,n:] = -tau**2*np.exp(-np.abs(d22)/lengthscale)/(lengthscale**2)
        #
        # elif nu == 1.5:
        #     wd11 = (3**0.5)*np.abs(d11)/lengthscale
        #     wd12 = (3**0.5)*np.abs(d12)/lengthscale
        #     wd22 = (3**0.5)*np.abs(d22)/lengthscale
        #
        #     # state observations
        #     a[:n,:n] = tau**2*(1+wd11)*np.exp(-wd11)
        #
        #     # cross terms
        #     a[n:,:n] = -3*tau**2*d12*np.exp(-wd12)/(lengthscale**2)
        #     a[:n,n:] = a[n:,:n].T
        #
        #     # derivative observations
        #     a[n:,n:] = 3*tau**2*(1-wd22)*np.exp(-wd22)/(lengthscale**2)
        #
        # elif nu == 2.5:
        #     wd11 = (5**0.5)*np.abs(d11)/lengthscale
        #     wd12 = (5**0.5)*np.abs(d12)/lengthscale
        #     wd22 = (5**0.5)*np.abs(d22)/lengthscale
        #
        #     # state observations
        #     a[:n,:n] = tau**2*(1+(wd11**2)/3+wd11)*np.exp(-wd11)
        #
        #     # cross terms
        #     a[n:,:n] = -5*tau**2*d12*(1+wd12)*np.exp(-wd12)/(3*lengthscale**2)
        #     a[:n,n:] = a[n:,:n].T
        #
        #     # derivative observations
        #     a[n:,n:] = -5*tau**2*(1-wd22**2+wd22)*np.exp(-wd22)/(3*lengthscale**2)
        #
        # else:
        wd11 = (nu**0.5)*np.abs(d11)/lengthscale
        wd12 = (nu**0.5)*np.abs(d12)/lengthscale
        wd22 = (nu**0.5)*np.abs(d22)/lengthscale

        # state observations
        a[:n,:n] = (2**(1-nu/2)/gamma(nu))*(wd11**nu)*kv(nu,(2**0.5)*wd11)
        # np.fill_diagonal(a[:n,:n],1+np.finfo('float64').eps**0.5)
        a[:n,:n] = tau**2*a[:n,:n]


        # cross terms
        a[n:,:n] = -(2**(1.5-nu/2)/(lengthscale**2*gamma(nu)))*nu*d12*(wd12**(nu-1))*kv(nu-1,(2**0.5)*wd12)
        a[n:,:n] *= tau**2
        a[:n,n:] = a[n:,:n].T

        # derivative observations
        a[n:,n:] = (2**(1-nu/2)/(lengthscale**3*gamma(nu)))*nu*(wd22**(nu-1))
        a[n:,n:] *= -2*tau**2*np.abs(d22)*(nu**0.5)*kv(nu-2,(2**0.5)*wd22) + (2**0.5)*lengthscale*kv(nu-1,(2**0.5)*wd22)

        return .5*(a+a.T)

    def __init__(self,nu=1.5,lengthscale=1.0,tau=1.0):
        if nu < 1.5:
            raise ValueError("nu must be at least 1.5 for the kernel to be differentiable")
        self.kernel_args = {'nu':nu,'lengthscale':lengthscale,'tau':tau}

# class DConstant(DKernel):
#     @staticmethod
#     def func(t,dt,c1,c2):
#         diag = np.concatenate((c1*np.ones_like(t),c2*np.ones_like(dt)))
#         return np.diag(diag)
#     def __init__(self,c1=1.0,c2=1.0):
#         self.kernel_args = {'c1':c1,'c2':c2}

class DWhiteKernel(DKernel):
    @staticmethod
    def func(t,dt,sigma1,sigma2):
        diag = np.concatenate((sigma1**2*np.ones_like(t),sigma2**2*np.ones_like(dt)))
        return np.diag(diag)
    def __init__(self,sigma1=1.0,sigma2=1.0):
        self.kernel_args = {'sigma1':sigma1,'sigma2':sigma2}

class DGaussianProcess:
    def __init__(self,kernel,mu=None,dmu=None):
        self.kernel = kernel
        if (mu is not None and dmu is None) or (mu is None and dmu is not None):
            raise ValueError("both mu and dmu must be provided if one is provided")
        self.mu = mu
        self.dmu = dmu

    def rvs(self,t,dt,size=1):
        n,m = len(t),len(dt)
        cov = self.kernel(t,dt)
        out = mvn.rvs(cov=cov,size=size)
        y,dy = out[:n],out[n:]
        if self.mu:
            y += self.mu(t)
            dy += self.dmu(dt)
        return y,dy

    def __call__(self,*args,**kwargs):
        return self.rvs(*args,**kwargs)

class DGaussianProcessRegressor:
    def __init__(self,kernel,mu=None,dmu=None):
        self.kernel = kernel
        if (mu is not None and dmu is None) or (mu is None and dmu is not None):
            raise ValueError("both mu and dmu must be provided if one is provided")
        self.mu = mu
        self.dmu = dmu

    def fit(self,t,dt,y,dy):
        self.t_train = t
        self.dt_train = dt
        self.y_train = y
        self.dy_train = dy
        n,m = len(t),len(dt)
        cov = self.kernel(t,dt)
        self.c,self.lower = la.cho_factor(cov)
        b = np.empty(n+m)
        b[:n],b[n:] = y,dy
        if self.mu:
            mu1 = self.mu(t)
            mu2 = self.dmu(dt)
            b[:n] -= mu1
            b[n:] -= mu2
        self.alpha = la.cho_solve((self.c,self.lower),b)

    def predict(self,t,dt,return_cov=False):
        n,m = len(t),len(dt)
        n_t,m_t = len(self.t_train),len(self.dt_train)
        if not hasattr(self,'t_train'):
            if self.mu:
                mu_hat = np.empty(n+m)
                mu_hat[:n],mu_hat[n:] = self.mu(t),self.dmu(t)
            else:
                mu_hat = np.zeros(n+m)

            cov_hat = self.kernel(t,dt)
        else:
            cov = self.kernel(np.concatenate((self.t_train,t)),
                              np.concatenate((self.dt_train,dt)))
            # build matrices
            cov12 = np.empty((n_t+m_t,n+m))
            cov12[:n_t,:n] = cov[:n_t,n_t:n_t+n]
            cov12[:n_t,n:] = cov[:n_t,-m:]
            cov12[n_t:,:n] = cov[n_t+n:-m,n_t:n_t+n]
            cov12[n_t:,n:] = cov[n_t+n:-m,-m:]
            cov22 = np.empty((n+m,n+m))
            cov22[:n,:n] = cov[n_t:n_t+n,n_t:n_t+n]
            cov22[:n,n:] = cov[n_t:n_t+n,-m:]
            cov22[n:,:n] = cov[-m:,n_t:n_t+n]
            cov22[n:,n:] = cov[-m:,-m:]

            mu_hat = cov12.T@self.alpha

            if self.mu:
                mu_hat[:n] += self.mu(t)
                mu_hat[n:] += self.dmu(dt)

            if return_cov: cov_hat = cov22 - cov12.T@la.cho_solve((self.c,self.lower),cov12)

        if return_cov: return mu_hat[:n], mu_hat[n:], cov_hat
        else: return mu_hat[:n], mu_hat[n:]

    def rvs(self,t,dt,size=1):
        n,m = len(t),len(dt)
        y, dy, cov = self.predict(t,dt,return_cov=True)
        out = mvn.rvs(cov=cov,size=size)
        return out[:n] + y, out[n:] + dy
