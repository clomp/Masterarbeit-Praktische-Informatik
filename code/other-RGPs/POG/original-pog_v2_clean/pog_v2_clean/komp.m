%komp.m

function [idxkomp,Wkomp] = komp(D,W,kernel,eps2,opt)

% see if the kernel matrix / inverse is supplied and build it if not
if isfield(opt,'KDD')
    KDD = opt.KDD; 
else
    KDD = kernel.f(D,D);
    KDD = KDD + 1e-9*eye(size(KDD));
end
if isfield(opt,'KDDinv')
    KDDinv = opt.KDDinv;
else
    KDDinv = inv(KDD);
end
k11 = KDD(1,1);

% initialization and bookkeeping
Rkomp = W;
Ikomp = false(size(D,2));
Kkompinv = [];

% build approximation one element at a timeB
for l=1:size(D,2)
    % find atom most colinear with residual
    Alpha = Rkomp*KDD;
    ips = sum(Alpha);
    [maxip,maxidx] = max(ips);

    % add this atom to the dictionary and compute projection
    Ikomp(maxidx) = true;
    Bkomp = W*KDD(:,Ikomp)/KDD(Ikomp,Ikomp);
    Rkomp = W;
    Rkomp(:,Ikomp) = Rkomp(:,Ikomp) - Bkomp;

    % test stopping criterion
    gamma = sum(Rkomp.*(Rkomp*KDD),2);
    if gamma<eps2
        idxkomp = Ikomp;
        Wkomp = Bkomp;
    end
end
