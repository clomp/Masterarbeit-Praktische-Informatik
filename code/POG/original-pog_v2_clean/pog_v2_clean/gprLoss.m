%gprLoss.m
%
%DESCRIPTION:
%    computes negative log-likelihood associated with Gaussian process
%    regression model
%    the likelihood function is
%    P( y | x, \ccalD, \bbw ) = 1/\sqrt{ (2 pi)^M |Sigma|}\times 
%       \exp\{ -(1/2)\|y - f(x) \|^2_{\Sigma^{-1}} \} 
%    and gradients
%
%   K_{D,D} is the kernel matrix parameterizing f
%INPUTS:
%    X: data matrix to evaluate against
%    y: class labels in {1,...,C}
%    D: current dictionary
%    W: current classifier weights
%    kernel: kernel function
%    lambda: 2-norm regularization parameter
%
%OUTPUTS:
%    l: Gaussian negative log likelihood value
%    dW: gradient of classifier weights corresponding to current dictionary
%    dWnew: gradient of classifier weights corresponding to new data
%    dtheta: gradient of kernel parameters

function [l,dW,dWnew,dtheta] = gprLoss(X,y,D,W,kernel,lambda)

% define \alpha in loss function
%alpha = 1.0;
%cross validation batch size = 500
% compute loss
K_DD = kernel.f(D,D); %compute kernel matrix
K_DD = K_DD + 1e-2*eye(size(K_DD)); %regularize by tiny constant
K_XX = kernel.f(X,X); %compute kernel matrix associated to new data
K_XX= K_XX + 1e-2*eye(size(K_XX)); %regularize by tiny constant
K = kernel.f(D,X); %compute empirical kernel map
%K=K+ 1e-1*ones(size(K));
%W=W(1,:);
n=length(K); %batch size
f_of_x= W*K_DD; %pseudo-inputs is a vector of length m, m is batch size;
muuu= K'*K_DD*f_of_x';
sigma=1e-2;
covariance=diag(diag(K_XX - K'*K_DD^(-1)*K)) + sigma^2*eye(length(y));


%m=length(K_DD); %model order
%idxclass = sub2ind([size(L,1) size(L,2)],y,1:size(L,2));
%Lclass = L(idxclass);
l=   ( + (n/2)*log(2* pi) + (1/2)*log(det(covariance)) ...
        +(1/2)*(y-muuu')*covariance^(-1)* (y-muuu')') ...
        + lambda*sum(sum(W.*(kernel.f(D,D)*W.').'));
    

 %   sigma_sq=K_XX - K'*K_DD^(-1)*K +sigma^2;
%likelihood=(2*pi*sigma_sq)^(-1/2)*exp(-(1/2)*(y-muuu)^2/sigma_sq);


% functional gradient terms
if nargout>2
    % classifier update for existing terms (i.e., dictionary points D)
    dW = 2*lambda*W;

    % classifier update for new terms (i.e., dictionary points X)
   % dWnew = zeros(size(W,1),size(X,2));
   dWnew=2*(y - muuu')*covariance^(-1) .*(K'*K_DD*f_of_x')'/n;
    %dWnew(idxclass) = -1/size(X,2);
    %dWnew(idxmax) = 1/size(X,2);
    %dWnew(:,(alpha+Lmax-Lclass)<0) = 0;

    % kernel parameter update
    if isfield(kernel,'theta')
        dtheta = zeros(size(kernel.theta));
        for j=1:length(dtheta)
            dKdtj = kernel.dt{j}(K,D,X);
            dtheta(j) = sum(sum(dWnew.*(W*kernel.dt{j}(K,D,X)))) + ...
                            lambda*sum(sum(W.*(kernel.dt{j}(K_DD,D,D)*W.').'));
        end
    else
        dtheta = [];
    end
end
