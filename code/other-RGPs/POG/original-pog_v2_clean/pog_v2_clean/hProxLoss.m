%hProxLoss.m
%
%DESCRIPTION:
%    provides interface to the loss for the proximal problem in the Hilbert space, i.e.,
%        g = sum_c 1/(2eta)*(w_c*KDD*w_c - 2*w_c*KDDaug*waug_c + waug_c*KDaugDaug*waug_c)^2
%
%INPUTS:
%    *Daug: augmented dictionary
%    *Waug: augmented classification matrix (usually Waug is formed using eta)
%    *kernel: kernel struct
%    *eta: weighting parameter (usually a step size)
%
%OUTPUTS:
%    *hpl: struct that contains function handles that accept a current (D,W) and compute:
%        *g: the loss
%        *dD: the gradient of g wrt D
%        *dW: the gradient of g wrt W

function hpl = hProxLoss(Daug,Waug,kernel,eta)
    hpl.g = @(D,W) proxLossg(D,W,Daug,Waug,kernel,eta);
    hpl.dD = @(D,W) proxLossdD(D,W,Daug,Waug,kernel,eta);
    hpl.dW = @(D,W,KDD,KDDaug) proxLossdW(D,W,Daug,Waug,KDD,KDDaug,eta);
end

function g = proxLossg(D,W,Daug,Waug,kernel,eta)
% compute kernel matrices
    KDD = kernel.f(D,D);
    KDD = KDD + 1e-9*eye(size(KDD));
    KDDaug = kernel.f(D,Daug);
    KDaugDaug = kernel.f(Daug,Daug);
    KDaugDaug = KDaugDaug + 1e-9*eye(size(KDaugDaug));

    % compute loss
    g = 0;
    for c=1:size(W,1)
        g = g + 1/2/eta*(...
            W(c,:)*(KDD*W(c,:).') + ...
            -2*W(c,:)*(KDDaug*Waug(c,:).') + ...
            Waug(c,:)*(KDaugDaug*Waug(c,:).') );
    end
end

function dD = proxLossdD(D,W,Daug,Waug,kernel,eta)
% compute kernel matrices
    KDD = kernel.f(D,D);
    KDD = KDD + 1e-9*eye(size(KDD));
    KDDaug = kernel.f(D,Daug);

    % compute gradient wrt 
    dD = zeros(size(D));
    for c=1:size(W,1)
        A1 = W(c,:).' * W(c,:);
        A2 = W(c,:).' * Waug(c,:);
        dD = dD + 1/eta*(0.5*kernel.dfsym(A1,KDD,D) - kernel.df1(A2,KDDaug,D,Daug));
    end
end


function dW = proxLossdW(D,W,Daug,Waug,KDD,KDDaug,eta)
    % compute gradient wrt W
    dW = zeros(size(W));
    for c=1:size(W,1)
        dW(c,:) = 1/eta*( (KDD*W(c,:).') - (KDDaug*Waug(c,:).') ).';
    end
end
