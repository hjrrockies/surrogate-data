import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import sklearn.gaussian_process as gp
import scipy.linalg as la
from scipy.stats import gaussian_kde

def rw_metropolis_hastings(f,llh,lpr,cov,x0,n,burn_in,update=50,verbose=False,debug=False):
    X = [x0]
    y = f(x0)
    loglikelihood = [llh(y)]
    logprior = [lpr(x0)]
    accepted = 0
    while len(X) < n+burn_in:
        # update proposal covariance
        if update:
            if len(X) < burn_in and not len(X)%update:
                cov = np.cov(X,rowvar=False)

        # propose new parameters
        u = X[-1] + stats.multivariate_normal.rvs(cov=cov)

        # evaluate prior
        lpr_u = lpr(u)
        if lpr_u > np.NINF:
            # evaluate forward model
            y_u = f(u)

            # evaluate likelihood and prior
            llh_u = llh(y_u)

            logalpha = llh_u + lpr_u - loglikelihood[-1] - logprior[-1]
        else:
            logalpha = np.NINF

        # metropolis-hastings accept/reject
        if np.log(np.random.rand()) < logalpha:
            X.append(u)
            y = y_u
            loglikelihood.append(llh_u)
            logprior.append(lpr_u)
            if len(X) > burn_in: accepted += 1
        else:
            X.append(X[-1])
            loglikelihood.append(loglikelihood[-1])
            logprior.append(logprior[-1])
        if verbose and not len(X)%1000:
            print(len(X))

    print("acceptance rate:",accepted/n)
    if debug:
        return np.array(X),np.array(logprior),np.array(loglikelihood),accepted/n
    else: return np.array(X[burn_in:])

def generate_data(f,y0,params,t0,tf,n_obs,sigma,nonneg=False):
    T_obs = np.linspace(t0,tf,n_obs)
    y_true = f(T_obs,y0,params)
    y_noisy = y_true + sigma*np.random.randn(y_true.shape[0],y_true.shape[1])
    if nonneg:
        y_noisy = np.abs(y_noisy)
    return T_obs,y_noisy,y_true

def fit_gp(y_noisy,T_obs):
    # create & fit gp
    kernel = gp.kernels.ConstantKernel()*gp.kernels.RBF()+gp.kernels.WhiteKernel()
    y_gp = gp.GaussianProcessRegressor(kernel)
    y_gp = y_gp.fit(T_obs[:,np.newaxis],y_noisy.T)
    return y_gp

def gp_data(y_gp,t0,tf,n_sur):
    # get gp mean and covariance for surrogate data
    T_sur = np.linspace(t0,tf,n_sur)
    mean,cov = y_gp.predict(T_sur[:,np.newaxis],return_cov=True)

    # symmetrize
    cov = .5*(cov+cov.T)

    # eigendecomposition of covariance -> precision matrix
    e,v = la.eig(cov)
    W = np.real(np.diag(np.sqrt(1/e))@(v.T))

    return T_sur,mean,cov,W

def build_kdes(data):
    kdes = []
    for col in data.T:
        kdes.append(gaussian_kde(col))
    return kdes
