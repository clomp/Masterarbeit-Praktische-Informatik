%gprPerf.m
%
%DESCRIPTION:
%    puts Minotaur's walk data into a form for Gaussian process regression
%INPUTS:

function data = legged_data_produce(ntrain,ncv,ntest)

data0=xlsread('RandomWalks.xlsx');
idx=randperm(size(data0,1));
data0=data0(idx,:);
feature_vectors=data0(:,1:4);
targets=data0(:,10);

data={};
data.Xtrain=feature_vectors(1:ntrain,:)';
data.ytrain=targets(1:ntrain)';

data.Xcv=feature_vectors(ntrain+1:ntrain+ncv,:)';
data.ycv=targets(ntrain+1:ntrain+ncv)';

data.Xtest=feature_vectors(ntrain+ncv + 1 : ntrain+ncv+ntest,:)';
data.ytest=targets(ntrain+ncv + 1 : ntrain+ncv+ntest)';


end