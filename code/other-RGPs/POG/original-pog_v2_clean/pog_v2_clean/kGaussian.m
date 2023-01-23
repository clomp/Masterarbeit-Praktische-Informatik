function kern = kGaussian(sig);
kern.f   = @(x,y)     kernel(sig, x, y);
kern.df1 = @(A,K,x,y) ddkernel(sig, A, K, x, y);
end

% compute kernel over a multi-input pair
function K = kernel(sig, X, Y)
N1 = size(X,2);
N2 = size(Y,2);
K = exp( -distEucSq(X,Y) / (2*sig^2) );
end

% compute the directional derivative of the kernel matrix along A
% conceptually:
%
% dX(i,j) = <A, dK/dX_ij>
%
% but since only the jth row of K depends on the jth column of X1,
%
% <A, dK/dX_ij> => <A.j, dK.j/dX_ij>
%
% and if we can express the derivative of the jth row with respect to
%  the jth column, then:
%
% dX(:,j) = [dK.j/dX_j]*A.j'
%
function dX = ddkernel(sig, A, K, X, Y)
D = size(X,1);
[N1 N2] = size(K);
dX = zeros(size(X));
for j = 1:N1
  %       dK_jk/dX_j       = (1/sig^2)*K(j,k)*(Y_k - X_j)
  %   =>  dK.j /dX_j       = (1/sig^2)*bsxfun(@minus, Y, X_j)*diag(K.j)
  %   => [dK.j /dX_j]*A.j' = (1/sig^2)*bsxfun(@minus, Y, X_j)*(K.j .* A.j)'
    dX(:,j) = (1/sig^2)*bsxfun(@minus,Y,X(:,j))*(K(j,:).*A(j,:))';
end

end

% http://www.cs.columbia.edu/~mmerler/project/code/pdist2.m
function D = distEucSq( X, Y )
m = size(X,2); n = size(Y,2);
XX = sum(X'.*X',2);
YY = sum(Y.*Y,1);
D = XX(:,ones(1,n)) + YY(ones(1,m),:) - 2*X'*Y;
end
