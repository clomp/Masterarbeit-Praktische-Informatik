%% construct approximations to the kernel matrix
% --random Fourier Features of Recht
% -- Nystrom approximation
%then do batch SVM or kernel logistic regression. 
%addpath(genpath(folder));
% clean up
clear all; close all; clc; drawnow;
prob='classification';
method='rff';
%********************************************************************

%********************************************************************
% [script version: load up pre-generated synthetic data]
 %data = load('mnist_data.mat');
 switch prob
     case 'regression'
        train=900; test=300; cv=133;
        data=legged_data_produce(train,cv,test);
        %loss = @gprLoss; % loss function
        %perf = @gprPerf;
        loss = @sqLoss; % loss function
        perf = @sqPerf;
     case 'classification'
         loss = @mcklrLoss; % loss function
         perf = @mcklrPerf;
% %%% loss = @mckhLoss;
% %%% perf = @mckhPerf;
% data = load('brodatz_data.mat');
 data = load('mnist_data.mat');
% data = load('multidist_data_c5m3_5k.mat');
% %%% data = load('multigauss_data3.mat');
% %%% data = load('multigauss_data5.mat');
% %%% data = load('multigauss_data10.mat');
% %%% data = load('multigauss_data22.mat');
% %%% data = load('multidist_data_c5m3.mat');
 end
ntrain = size(data.Xtrain,2);
param_dim=size(data.Xtrain,1);
%********************************************************************

% %%% %********************************************************************
% %%% % [script version: resort data]
% %%% [foo,idxsrt] = sort(data.ytrain);
% %%% data.Xtrain = data.Xtrain(:,idxsrt);
% %%% data.ytrain = data.ytrain(:,idxsrt);
% %%% %********************************************************************

%********************************************************************
% [script version: define loss and kernel function]
theta(1) = 1; % scalar multiple on kernel value
theta(2) = .6; % kernel bandwidth [mnist_data.mat]
%theta(2) = 6; % [brodatz_data.mat]
%theta(2) = 2.0*data.sig; % [multi*_data*.mat]
addpath(genpath(fileparts(which('btls_iter.m'))));


 kernel = kRBF(theta); % kernel function
% %%% kernel = kLinear;
% %%% kernel = kChiSquared;

switch method
    %% Random fourier features
    case 'rff'
p_samp=10000;  nsamples= ntrain; %nsamples=round(ntrain/3);
kobj = InitExplicitKernel('rbf',.6, param_dim, p_samp,[]);
%construct random fourier features for training set
z_omega_train = rf_featurize(kobj, double(data.Xtrain(:,1:nsamples))',p_samp);
%construct random fourier features for test set
z_omega_test = rf_featurize(kobj, double(data.Xtest)',p_samp);
labels_train=sparse(data.ytrain(:,1:nsamples)); data_train_rff=sparse(z_omega_train(1:nsamples,:) );
labels_test=sparse(data.ytest); data_test_rff=sparse(z_omega_test );
%addpath('/Users/aekoppel/Documents/MATLAB/Research Code/polk/POLK_1.R1/liblinear/matlab')
%SVM train
fourier_features_svm=train(labels_train',data_train_rff,'-s 2 -c 5 -B .05'); 

% and test
predict( labels_test', data_test_rff, fourier_features_svm);





%Logistic train
fourier_features_lr=train(labels_train',data_train_rff,'-s 0 -c 5 -B .05'); 
% and test
predict( labels_test', data_test_rff, fourier_features_lr);

    case 'nystrom'
     %% Nystrom approximation of kernel matrix
      nsamples= round(ntrain/3);
% usps 38 data set;
%load('usps38.mat'); 
m=1000; MaxIter=50;
% average squared distance
b=.6;
% construct an rbf kernel in the form of exp(-||x||^2/b);
kernel = struct('type', 'rbf', 'para', b); 

% RBF kernel with kernel width b
%K = exp(-sqdist(data.Xtrain, data.Xtrain)/b);

%for i = 1:10;
    %m = i*10;
    Kt1_train = INys(kernel,data.Xtrain(:,1:nsamples)', m, 'k'); %option $k$ means k-means with m points
    
    Kt1_test = INys(kernel,data.Xtest', m, 'k');
    
    labels_train=sparse(data.ytrain(:,1:nsamples)); 
    labels_test=sparse(data.ytest); 
    
    data_nystrom_kmeans=sparse(Kt1_train); 
    data_nystrom_kmeans_test=sparse(Kt1_test); 
    
    nystrom_svm=train(labels_train',data_nystrom_kmeans,'-c 100 -B .031'); 
    predict( labels_test', data_nystrom_kmeans_test, nystrom_svm);
    
% Kt2_train = INys(kernel,data.Xtrain', m, 'r'); %option $r$ means use random m points
%     Kt2_test = INys(kernel,data.Xtest', m, 'r'); %option $r$ means use random m points
%     data_nystrom_rand_test=sparse(Kt2_test);
%     data_nystrom_rand=sparse(Kt2_train);
%     
end