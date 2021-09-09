import numpy as np
from scipy.integrate import solve_ivp
from scipy.stats import gaussian_kde
from sur_data import *
import matplotlib.pyplot as plt

def lotka_volterra(t,y,a,b,c,d):
    dy = np.empty_like(y)
    # switched to match
    dy[0] = d*y[0]*y[1]-c*y[0]
    dy[1] = a*y[1]-b*y[0]*y[1]
    return dy

def solve_lv(T,y0,theta):
    sol = solve_ivp(lotka_volterra,(T[0],T[-1]),y0,t_eval=T,args=theta)
    return sol.y

# uniform prior
def lpr(x):
    if np.all((0<x)&(x<10)):
        return 0
    else:
        return np.NINF

# Observational model and likelihood (standard inference)
def f_obs(theta):
    return solve_lv(T_obs,theta[:2],[1,theta[2],theta[3],1])

def llh_obs(y):
    # least squares
    return -((y-y_noisy)**2).sum()

# Surrogate model and likelihood (unweighted)
def f_sur(theta):
    return solve_lv(T_sur,theta[:2],[1,theta[2],theta[3],1])

def llh_sur(y):
    return -((y-mean.T)**2).sum()

# Surrogate likelihood (precision matrix weighted)
def llh_wsur(y):
    return -la.norm(W@(y[0]-mean[:,0]))**2 - la.norm(W@(y[1]-mean[:,1]))**2

if __name__=="__main__":
    n_samp = 1000000
    burn_in = 20000
    y0 = [2,.5]
    theta = np.array([y0[0],y0[1],1,1])
    params = np.array([1,1,1,1])
    t0,tf = 0,10
    sigma = 0.02
    for n_obs in [5,10,20]:
        T_obs,y_noisy,y_true = generate_data(solve_lv,y0,params,t0,tf,
                                             n_obs,sigma,nonneg=True)
        fig = plt.figure()
        plt.plot(T_obs,y_true[0])
        plt.scatter(T_obs,y_noisy[0])
        plt.title('True solution and noisy data')
        plt.show()
        np.savez(f'data/lv_data_nobs{n_obs}.npz',T_obs=T_obs,y_noisy=y_noisy,y_true=y_true)

        prop_cov = .03*np.eye(4)
        Theta,LPR,LLH,ar = rw_metropolis_hastings(f_obs,llh_obs,lpr,prop_cov,theta,
                                       n=n_samp,burn_in=burn_in,update=False,
                                       verbose=False,debug=True)
        fig = plt.figure()
        plt.hist(Theta[:,0],density=True)
        plt.title('samples histogram')
        plt.show()
        np.savez(f'data/lv_std_nobs{n_obs}.npz',Theta=Theta[burn_in:],lpr=LPR[burn_in:],
                 llh=LLH[burn_in:],ar=ar)

        y_gp = fit_gp(y_noisy,T_obs)
        for n_sur in [100,200,400]:
            T_sur,mean,cov,W = gp_data(y_gp,t0,tf,n_sur)
            np.savez(f'data/lv_sur_data_nobs{n_obs}_nsur{n_sur}.npz',T_sur=T_sur,mean=mean,cov=cov,W=W)

            # unweighted
            prop_cov = .002*np.eye(4)
            Theta,LPR,LLH,ar = rw_metropolis_hastings(f_sur,llh_sur,lpr,prop_cov,theta,
                                           n=n_samp,burn_in=burn_in,update=False,
                                           verbose=False,debug=True)
            np.savez(f'data/lv_sur_nobs{n_obs}_nsur{n_sur}.npz',Theta=Theta[burn_in:],lpr=LPR[burn_in:],
                     llh=LLH[burn_in:],ar=ar)

            # precision matrix weighted
            Theta,LPR,LLH,ar = rw_metropolis_hastings(f_sur,llh_wsur,lpr,prop_cov,theta,
                                           n=n_samp,burn_in=burn_in,update=False,
                                           verbose=False,debug=True)
            np.savez(f'data/lv_wsur_nobs{n_obs}_nsur{n_sur}.npz',Theta=Theta[burn_in:],lpr=LPR[burn_in:],
                     llh=LLH[burn_in:],ar=ar)
