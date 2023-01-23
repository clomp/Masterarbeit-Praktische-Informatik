function kern = kChiSquared;
kern.f   = @kernel;
kern.df1 = @ddkernel;
kern.dfsym = @ddkernelsym;
end

% compute kernel over a multi-input pair
function K = kernel(X, Y)
% pre-allocate kernel matrix
    K = zeros(size(X,2),size(Y,2));
    
    % compute kernel matrix row by row
    for j=1:size(X,2)
        P = bsxfun(@times,X(:,j),Y);
        S = bsxfun(@plus,X(:,j),Y);
        K(j,:) = 2*sum(P./max(S,eps));
    end
    
% $$$     % compute kernel matrix row by row
% $$$     for j=1:size(X,2)
% $$$         Diff = bsxfun(@minus,X(:,j),Y);
% $$$         Sum = bsxfun(@plus,X(:,j),Y);
% $$$ 
% $$$         % Sum could be zero, but when it is, Diff should also be
% $$$         % zero. adjust these matrices such that the corresponding term in
% $$$         % the sum will yield zero instead of NaN
% $$$         Sum(Sum<eps)=1;
% $$$         Diff(Sum<eps)=0;
% $$$ 
% $$$         K(j,:) = 1 - 0.5*sum(bsxfun(@rdivide,Diff.^2,Sum),1);
% $$$     end
end

% compute the directional derivative of the kernel matrix along A
% conceptually:
%
% dX(i,j) = <A, dK/dX_ij>
%
% but since only the jth row of K depends on the jth column of X,
%
% <A, dK/dX_ij> => <A.j, dK.j/dX_ij>
%
% and if we can express the derivative of the jth row with respect to
%  the jth column, then:
%
% dX(:,j) = [dK.j/dX_j]*A.j'
%
function dX = ddkernel(A, K, X, Y)
% pre-allocate derivative matrix
    dX = zeros(size(X));

    % compute column by column according to the above
    for j=1:size(X,2)
        S = bsxfun(@plus,X(:,j),Y);
        dX(:,j) = ( 2*(Y.^2)./(max(S.^2,eps)) ) * (A(j,:).');
    end
end

% comput the directional derivative of the kernel matrix along A *when* the kernel matrix is KXX
% for the first argument.  for the chi-squared kernel, this yields a special case when X_ij = 0!
function dX = ddkernelsym(A, K, X)
% pre-allocate derivative matrix
    dX = zeros(size(X));
    
    % compute column by column according to the above
    for j=1:size(X,2)
        S = bsxfun(@plus,X(:,j),X);
        G = 2*(X.^2 ./ max(S.^2, eps));
        G(:,j) = 0.5;
        dX(:,j) =  2*G * (A(j,:).');
    end
end
