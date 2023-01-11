# -*- coding: utf-8 -*-
# AUTHOR 	Marco Huber, 2012, marco.huber@ieee.org
#% RECURSIVEGP Gaussian process regression by means of a fixed set of basis vectors.
#
# IN    x           Location of the basis vectors
#       m           Vector of means of latent function at the basis vectors
#       C           Covariance of latent function at the basis vectors
#       xn          New inputs
#       yn          New observations. If empty, merely the mean and
#                   covariance at the new inputs 'xn' is predicted and no
#                   update of the basis vectors is performed.
#       meanfun     Mean function
#       covfun      Covariance function
#       hyp         Hyper-parameters of mean and covariance functions
#       invKernel   Inverted kernel matrix for the basis vectors. If empty,
#                   the next argument 'kernel' has to be set.
#       kernel      Kernel matrix of the basis vectors. Only used if
#                   'invKernel' is empty.
#
# OUT   m           Updated mean of latent function or predicted mean at
#                   input 'xn' if 'yn' is empty.
#       C           Updated covariance of latent function or predicted
#                   covariance if input 'xn' if 'yn' is empty.
#
import numpy as np
	
def recursiveGP(x, m, C, xn, yn, meanfun, covfun, hyp, invKernel=[], Kernel=[]):

    Nn = yn.shape[0];   #Nn = length(yn);
    Cnoise = np.exp(2*hyp['lik']); #Cnoise = exp(2*hyp.lik); % measurement noise variance

    # Kernels and means
    Knx = covfun(hyp['cov'], xn,x)          # Knx = feval(covfun{:}, hyp.cov, xn, x);
    Kn = covfun(hyp['cov'], xn)             # Kn = feval(covfun{:}, hyp.cov, xn);
    mn = meanfun(hyp['mean'],xn)            # mn = feval(meanfun{:}, hyp.mean, xn);
    mx = meanfun(hyp['mean'], x);           # mx = feval(meanfun{:}, hyp.mean, x);

    if(invKernel==[] and Kernel==[]):       # if nargin <= 8 || (isempty(invKernel) && isempty(Kernel))
        Kernel = covfun(hyp['cov'],x)       # Kernel = feval(covfun{:}, hyp.cov, x);
        print('Neither inv. kernel nor kernel parameter is present. Kernel calculated, but for better on-line performance provide pre-computed kernel or inv. kernel.')

 
    if(invKernel==[]):      # if isempty(invKernel)
                           	# %Kxx = feval(covfun{:}, hyp.cov, x);
	    J = Knx @ np.linalg.inv(Kernel)      # J = Knx/Kernel;
    else:
	    J = Knx @ invKernel    # J = Knx*invKernel;

    D = Kn - J @ Knx.T     # D = Kn - J*Knx';
    
    # % Covariances of new data
    Cxn = C @ J.T           # Cxn = C*J';
    Cn = D + J @ Cxn        # Cn = D + J*Cxn;

    # % mean of new data
    mn = mn + J @ (m-mx)     # mn = mn + J*(m - mx);

    # % Predict values if no observations are available or 
    # % update basis vectors in case of observations
    if(yn==[]):      # if isempty(yn)
        m = mn
        C = Cn + Cnoise @ np.eye(Cn.shape[0])   # C = Cn + Cnoise*eye(size(Cn));
    else:
     # % Kalman Gain
        G = Cxn @ np.linalg.inv(Cn + Cnoise*np.eye(Nn))  # G  = Cxn/(Cn + Cnoise*eye(Nn));
    
    # % Update basis vectors
    m = m + G @ (yn - mn)        # m = m + G*(yn - mn);
    C = C - G @ Cxn.T           # C = C - G*Cxn';
    return(m, C)