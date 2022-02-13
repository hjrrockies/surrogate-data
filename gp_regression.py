import numpy as np
import scipy.linalg as la
from scipy.stats import multivariate_normal as mvn
from scipy.spatial.distance import pdist, squareform
from scipy.special import kv, gamma

class Kernel:
    def __init__(self,func,**kernel_args):
        self.func = func
        self.kernel_args = kernel_args

    def __call__(self,x):
        return self.func(x,**self.kernel_args)

    @classmethod
    def dist(cls,X):
        if X.ndim == 1:
            X = X[:,np.newaxis]
        return pdist(X)

    def __add__(self,other):
        def func(X,kernel_args1,kernel_args2):
            return self.func(X,**kernel_args1)+other.func(X,**kernel_args2)
        kernel_args = {'kernel_args1':self.kernel_args,'kernel_args2':other.kernel_args}
        return Kernel(func,**kernel_args)

    def __sub__(self,other):
        def func(X,kernel_args1,kernel_args2):
            return self.func(X,**kernel_args1)-other.func(X,**kernel_args2)
        kernel_args = {'kernel_args1':self.kernel_args,'kernel_args2':other.kernel_args}
        return Kernel(func,**kernel_args)

    def __mul__(self,other):
        def func(X,kernel_args1,kernel_args2):
            return self.func(X,**kernel_args1)*other.func(X,**kernel_args2)
        kernel_args = {'kernel_args1':self.kernel_args,'kernel_args2':other.kernel_args}
        return Kernel(func,**kernel_args)

    __rmul__ = __mul__

class SquareExp(Kernel):
    @staticmethod
    def func(X,lengthscale,tau):
        d = Kernel.dist(X)
        a = squareform(np.exp(-d**2/(2*lengthscale**2)))
        np.fill_diagonal(a,1+np.finfo('float64').eps**0.5)
        return tau**2*a

    def __init__(self,lengthscale=1.0,tau=1.0):
        self.kernel_args = {'lengthscale':lengthscale,'tau':tau}

class Matern(Kernel):
    @staticmethod
    def func(X,nu,lengthscale,tau):
        d = Kernel.dist(X)
        frac = (2*nu)**0.5*np.abs(d)/lengthscale
        a = squareform(2**(1-nu)/gamma(nu)*((frac)**nu)*kv(nu,frac))
        np.fill_diagonal(a,1+np.finfo('float64').eps**0.5)
        return tau**2*a

    def __init__(self,nu=1.5,lengthscale=1.0,tau=1.0):
        self.kernel_args = {'nu':nu,'lengthscale':lengthscale,'tau':tau}

class WhiteNoise(Kernel):
    @staticmethod
    def func(X,C):
        return C*np.eye(X.size)
    def __init__(self,C=1.0):
        self.kernel_args = {'C':C}

class Periodic(Kernel):
    @staticmethod
    def func(X,lengthscale,periodicity,tau):
        d = Kernel.dist(X)
        a = squareform(np.exp(-2*np.sin(np.pi*d/periodicity)**2/lengthscale**2))
        np.fill_diagonal(a,1+np.finfo('float64').eps**0.5)
        return tau**2*a
    def __init__(self,lengthscale=1.0,periodicity=1.0,tau=1.0):
        self.kernel_args = {'lengthscale':lengthscale,
                            'periodicity':periodicity,
                            'tau':tau}

class GaussianProcess:
    def __init__(self,kernel,mu=None):
        self.kernel = kernel
        self.mu = mu

    def rvs(self,X,n=1):
        cov = self.kernel(X)
        Y = mvn.rvs(cov=cov,size=n)
        if self.mu:
            Y += self.mu(X)
        return Y

    def __call__(self,*args):
        return self.rvs(*args)

class GaussianProcessRegressor(GaussianProcess):
    def fit(self,X,Y):
        self.X_train = X
        self.Y_train = Y
        cov = self.kernel(X)
        self.c,self.lower = la.cho_factor(cov)
        b = self.Y_train
        if self.mu:
            mu1 = self.mu(self.X_train)
            b -= mu1
        self.alpha = la.cho_solve((self.c,self.lower),b)

    def predict(self,X,ret_cov=False):
        if not hasattr(self,'X_train'):
            if self.mu:
                mu_hat = self.mu(X)
            else:
                mu_hat = np.zeros_like(X)

            cov_hat = self.kernel(X)
        else:
            cov = self.kernel(np.concatenate((X,self.X_train)))
            cov11 = cov[:X.size,:X.size]
            cov12 = cov[:X.size:,X.size:]

            mu_hat = cov12@self.alpha

            if self.mu:
                mu_hat += self.mu(X).flatten()

            if ret_cov: cov_hat = cov11 - cov12@la.cho_solve((self.c,self.lower),cov12.T)

        if ret_cov: return mu_hat, cov_hat
        else: return mu_hat

    def rvs(self,X,n=1):
        mu_hat, cov_hat = self.predict(X,ret_cov=True)
        Y = mvn.rvs(cov=cov_hat,size=n)
        return Y + mu_hat
