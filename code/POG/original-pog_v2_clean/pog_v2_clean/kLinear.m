function kern = kLinear
kern.f   = @(x,y)     kernel(x, y);
kern.df1 = @(A,K,x,y) ddkernel(A, K, x, y);
end

% compute kernel over a multi-input pair
function K = kernel(X, Y)
K = X.'*Y;
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
function dX = ddkernel(A, K, X, Y)
dX = Y*(A.');
end