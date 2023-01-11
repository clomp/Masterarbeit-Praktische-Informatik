function [m, C] = recursiveGP(x, m, C, xn, yn, meanfun, covfun, hyp, invKernel, Kernel)
% RECURSIVEGP Gaussian process regression by means of a fixed set of basis
% vectors.
%
% IN    x           Location of the basis vectors
%       m           Vector of means of latent function at the basis vectors
%       C           Covariance of latent function at the basis vectors
%       xn          New inputs
%       yn          New observations. If empty, merely the mean and
%                   covariance at the new inputs 'xn' is predicted and no
%                   update of the basis vectors is performed.
%       meanfun     Mean function
%       covfun      Covariance function
%       hyp         Hyper-parameters of mean and covariance functions
%       invKernel   Inverted kernel matrix for the basis vectors. If empty,
%                   the next argument 'kernel' has to be set.
%       kernel      Kernel matrix of the basis vectors. Only used if
%                   'invKernel' is empty.
%
% OUT   m           Updated mean of latent function or predicted mean at
%                   input 'xn' if 'yn' is empty.
%       C           Updated covariance of latent function or predicted
%                   covariance if input 'xn' if 'yn' is empty.
%
% AUTHOR 	Marco Huber, 2012, marco.huber@ieee.org
	

Nn = length(yn);
Cnoise = exp(2*hyp.lik); % measurement noise variance

% Kernels and means
Knx = feval(covfun{:}, hyp.cov, xn, x);
Kn = feval(covfun{:}, hyp.cov, xn);
mn = feval(meanfun{:}, hyp.mean, xn);
mx = feval(meanfun{:}, hyp.mean, x);

if nargin <= 8 || (isempty(invKernel) && isempty(Kernel))
    Kernel = feval(covfun{:}, hyp.cov, x);
    warning('Neither inv. kernel nor kernel parameter is present. Kernel calculated, but for better on-line performance provide pre-computed kernel or inv. kernel.');
    invKernel = [];
end

% 
if isempty(invKernel)
	%Kxx = feval(covfun{:}, hyp.cov, x);
	J = Knx/Kernel;
else
	J = Knx*invKernel;
end
D = Kn - J*Knx';
    
% Covariances of new data
Cxn = C*J';
Cn = D + J*Cxn;

% mean of new data
mn = mn + J*(m - mx);

% Predict values if no observations are available or 
% update basis vectors in case of observations
if isempty(yn)
    m = mn;
    C = Cn + Cnoise*eye(size(Cn));
else
    % Kalman Gain
    G = Cxn/(Cn + Cnoise*eye(Nn));
    
    % Update basis vectors
    m = m + G*(yn - mn);
    C = C - G*Cxn';
end