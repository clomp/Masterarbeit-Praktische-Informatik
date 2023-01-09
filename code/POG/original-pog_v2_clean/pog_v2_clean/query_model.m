function [ f_of_x, residual ] = query_model( X,y,T )
% simple function that evaluates the kernelized regressor $f(x) = \hat{y}$ 
% as well as the error residual $\hat{y} - y$ for a given mini-batch of data
% the dependencies are:
% the test points X, 
% the target variables for the mini-batch,
% T is how many data points we want to train the model for

%example inputs: %data.Xtest(:,tidxtest),
                 %data.Ytest(tidxtest)

[ D,W,kernel ] = soldd_function_quiet(T);


K = kernel.f(D,X) ; %compute empirical kernel map

f_of_x= W*K; %compute f

residual=f_of_x - y;


end

