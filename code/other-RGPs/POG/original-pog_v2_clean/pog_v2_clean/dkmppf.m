%dkmppf.m
%
%DESCRIPTION:
%    implements destructive version of kernel matching pursuit with pre-fitting (see [1])
%
%INPUTS:
%    *D: columns are dictionary points over which function(s) to approximate are defined
%
%    *W: each rows contains weights that define a particular function
%
%    *kernel: kernel struct
%
%    *eps2: squared-norm distance for stopping criterion
%
%    *opt: options struct with the following fields
%        *KDD: kernel matrix for D
%
%OUTPUTS:
%    *idxdkmppf: indices of dictionary points used (refer to columns of D)
%
%    *Wdkmppf: approximation dictionary weights for D(:,idxkomp)
%
%REFERENCES:
%    [1] Vincent and Bengio. "Kernel matching pursuit." Machine Learning. 2002.
%    [2] https://en.wikipedia.org/wiki/Woodbury_matrix_identity#Derivation_from_LDU_decomposition. accessed 19 May 2016.

function [idxdkmppf,Wdkmppf] = dkmppf(D,W,kernel,eps2,opt)

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

% the set of indices of D that we are keeping
Y = 1:size(D,2);
% the set of indices of D that we are removing
Z = [];
% the working copy of the growing matrix we need
Q = [];

% continue removing points as long as we have them to remove
continue_pruning = 1;
while continue_pruning
  % find the least-important element to delete
  gmin = Inf;
  for i = 1:length(Y)
      % removal set to consider
      Zi = [Z Y(i)];

      % grow the matrix
      mvi = KDDinv(Z,Y(i));
      mi = KDDinv(Y(i),Y(i));
      bi = Q*mvi;
      betai = 1/(mi-mvi.'*bi);
      qi = betai*bi;
      Qi = [Q+qi*bi.', -qi;-qi.', betai];

      % compute error for this removal set, see if smallest
      gi = sum(bsxfun(@dot,W(:,Zi).',Qi*(W(:,Zi).')));
      if gi<gmin
          gmin = gi;
          imin = i;
          Qmin = Qi;
      end
  end

  % if best error is still okay, delete the corresponding element
  if gmin <= eps2
    Z = [Z Y(imin)];
    Y(imin) = [];
    Q = Qmin;
  % otherwise, we are done
  else
    continue_pruning = 0;
  end
end

% return the indices that we kept
idxdkmppf = Y;
% project weights onto remaining indices
KYYinv = KDDinv(Y,Y) - KDDinv(Y,Z) * (Q*KDDinv(Z,Y));
Wdkmppf = W * KDD(:,Y) * KYYinv;
