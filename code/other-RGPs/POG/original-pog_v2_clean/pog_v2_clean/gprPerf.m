%gprPerf.m
%
%DESCRIPTION:
%    computes performance metrics for an mcklr model on a given batch of data
%
%INPUTS:
%    *X: columns are data points
%
%    *y: target variable, examples of unknown true process I(x)
%
%    *D: the dictionary (columns are points)
%
%    *W: Gaussian process mean regressor weights
%
%    *param: the omcklr parameters struct (contains, e.g., the kernel)
%
%OUTPUTS:
%    *perf: struct with the following fields
%        *pcorrect: percentage of samples correctly classified
%        *loss: loss function value

function perf = gprPerf(X,y,D,W,param)

% if -1 labels appear, we have +/-1 and should correct to 2/1
% if any(c==-1)
%     c = -0.5*c+1.5;
% end

% get sizes of some things
N = size(X,2);
%C = size(W,2)+1;

% compute per-class probabilities for each data point
K = param.kernel.f(D,X);
% L = [W.'*K;zeros(1,N)];
% Eta = exp(L);

K_DD = param.kernel.f(D,D); %compute kernel matrix
K_DD = K_DD + 1e-2*eye(size(K_DD)); %regularize by tiny constant
K_XX = param.kernel.f(X,X); %compute kernel matrix associated to new data
K_XX= K_XX + 1e-2*eye(size(K_XX)); %regularize by tiny constant
K = param.kernel.f(D,X); %compute empirical kernel map
%K = K + 1e-1*ones(size(K));
%W=W(1,:);
%n=length(K); %batch size
f_of_x= W'*K; %pseudo-inputs is a vector of length n, n is batch size;
sigma=1e-5;
covariance=diag(diag(K_XX - K'*K_DD^(-1)*K)) + sigma^2*eye(length(y));
perf=   ( + (n/2)*log(2* pi) + (1/2)*log(det(covariance)) ...
        +(1/2)*(y-muuu')*covariance^(-1)* (y-muuu')') ...
        + lambda*sum(sum(W.*(kernel.f(D,D)*W.').'));
%perf=norm( y- f_of_x)^2;
