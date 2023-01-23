%soldd.m
%
%DESCRIPTION:
%    performs parsimonious online gaussian process regression
%    described in [1]
%
%REFERENCES:
%    [1] A. Koppel, A. B. Singh, K. Rajawat. "Consistent Online Gaussian 
%    Process Regression  Without the Sample Complexity Bottleneck."
%    Submitted, 2018 NIPS.

%********************************************************************
% clean up
clear all; close all; clc; drawnow;
prob='regression';
%********************************************************************

%********************************************************************
% [script version: load up pre-generated synthetic data]
 %data = load('mnist_data.mat');
 switch prob
     case 'regression'
        train=900; test=300; cv=133;
     %   data=legged_data_produce(train,cv,test);
        data = load('Lidar_Data.mat');
        data=data.data;
        %loss = @gprLoss; % loss function
        %perf = @gprPerf;
        loss = @sqLoss; % loss function
        perf = @sqPerf;
     case 'classification'
         loss = @mcklrLoss; % loss function
         perf = @mcklrPerf;
% %%% loss = @mckhLoss;
% %%% perf = @mckhPerf;
% %%% data = load('brodatz_data.mat');
 data = load('multidist_data_c5m3_5k.mat');
% %%% data = load('multigauss_data3.mat');
% %%% data = load('multigauss_data5.mat');
% %%% data = load('multigauss_data10.mat');
% %%% data = load('multigauss_data22.mat');
% %%% data = load('multidist_data_c5m3.mat');
 end
ntrain = size(data.Xtrain,2);
%********************************************************************
% %%% %********************************************************************

%********************************************************************
% [script version: define loss and kernel function]
theta(1) = 1; % scalar multiple on kernel value
%theta(2) = 6; % for legged robot data
theta(2) = .01; % for LIDAR data
noise_prior=.15;  % for LIDAR data
%noise_prior=1;  % for legged robot data
 kernel = kRBF(theta); % kernel function
% %%% kernel = kLinear;
% %%% kernel = kChiSquared;




%********************************************************************

%********************************************************************
% [script version: define parameters]
T = 5*ntrain; %number of training samples to process (wraparound if large)

%Keps = 5*1e-3; % eps_h, parsimony constant, determines compression stringency
%********************************************************************
%good results for eta0=2e-1; Keps=50; %%% eta0=1e-1; Keps=70;
%********************************************************************
% initialize dictionary and classifier
M = zeros(1,T);
M(1) = 1;
D = data.Xtrain(:,1);
switch prob
    case 'regression'
       % W = 10^(-4)*randn(1,M(1));  
       W = 10^(-2)*randn(1,M(1));
    case 'classification'
        W = zeros(data.C,M(1));
end
theta1 = zeros(1,T);
theta1(1) = theta(1);
theta2 = zeros(1,T);
theta2(1) = theta(2);

% [debug] initialize time origin for dictionary
torigin = 1;
y=0; 
% [debug] track dictionary change
Dbinary = zeros(1,T);
Dangle = zeros(1,T);
mu_posterior=zeros(1,T);
Sigma_posterior=zeros(1,T);
eval_estimate=zeros(1,ntrain);
%********************************************************************

%********************************************************************
% begin online processing
fprintf('Beginning online optimization...\n');
lcv=zeros(1,T);
hellinger=zeros(1,T); bhattacharyya=zeros(1,T);
for t=(M(1)+1):T
    % handle indexing
    tidx = t:min(T,t);

    % handle possible wraparound
    tidxtrain = mod(tidx-1,ntrain)+1;

% %%%     % [debug: grabber for if training point is already in the dictionary]
% %%%     if any(ismember(tidxtrain,torigin))
% %%%         fprintf('\tREPEAT!\n');
% %%%     end


toriginold = torigin;



Dold=D;    
%Wold=W;
yold=y;

Daug=[D data.Xtrain(:,tidxtrain)]; 

    k_DX=kernel.f(Daug,data.Xtrain(:,tidxtrain));
    k_XX=kernel.f(data.Xtrain(:,tidxtrain),data.Xtrain(:,tidxtrain));
    KDaugDaug=kernel.f(Daug,Daug); %KDD = KDaugDaug;
    yaug=[y data.ytrain(tidxtrain)];
   Maug = size(Daug,2); 
    mu_posterior(t)=k_DX'/(KDaugDaug + noise_prior^2*eye(size(KDaugDaug)))*yaug';
    
 Sigma_posterior(t)=k_XX-k_XX*k_DX'/(KDaugDaug + noise_prior^2*eye(size(KDaugDaug)))*k_DX + noise_prior^2;   
    
    toriginaug = [toriginold tidx];

            D = Daug;
            y = yaug;
            KDD = KDaugDaug;
            torigin = toriginaug;
            M(tidx) = Maug;
       %     break;
    
    
    eval_estimate(tidxtrain)=mu_posterior(tidx);
    lcv(tidx) = norm(eval_estimate - data.ytrain)^2;
    
    % [debug] report
    fprintf('\titerate %d: M=%d, theta2=%.3e, ||D||=%.3e, ||W||=%.3e, rcond(K)=%.3e, lcv=%.3e\n', ...
            t,M(t),theta2(t),norm(D,'fro'),norm(W,'fro'),rcond(KDD),lcv(t));
end
%********************************************************************


%********************************************************************
% [script version: evaluate]



% report to console
%fprintf('\n\n');
fprintf('Dictionary size: %d (%d possible)\n',size(D,2),T);
%fprintf('Percent correct (train): %f\n',perftrain.pcorrect);
%fprintf('Percent correct (test): %f\n',perftest.pcorrect);
%% Figures
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plot(lcv(1:t),'LineWidth',2);%title('Training Loss');
 xlabel('$t$, number of samples','interpreter','Latex','FontSize',30);
 ylabel('Posterior Mean Square Error','interpreter','Latex','FontSize',30);
 set(gca,'FontSize',30);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 figure;
plot(hellinger(1:t),'LineWidth',2);%title('Training Loss');
%plot(bhattacharyya(1:t),'LineWidth',2);
%str{1}='Hellinger'; str{2}='Bhattacharyya'; 
%legend(str,'Location','Best','interpreter','Latex','FontSize',30);
 xlabel('$t$, number of samples','interpreter','Latex','FontSize',30);
 ylabel('Hellinger Distance','interpreter','Latex','FontSize',30); set(gca,'FontSize',30);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%       
    if size(data.Xtrain,1)==1
        figure;
        hold on; x =data.Xtrain;
        plot(x,data.ytrain,'LineWidth',1.5 );
        plot(x,mu_posterior(T-ntrain+1:end),'LineWidth',1.5,'LineStyle','--'); 
        xlabel('Range','interpreter','Latex','FontSize',30)
        ylabel('Log Ratio','interpreter','Latex','FontSize',30); 
        curve1 = mu_posterior(T-ntrain+1:end)+3*Sigma_posterior(T-ntrain+1:end);
        curve2 = mu_posterior(T-ntrain+1:end)-3*Sigma_posterior(T-ntrain+1:end);
        plot(x, curve1, 'r', 'LineWidth', 1);
        plot(x, curve2, 'b', 'LineWidth', 1);
        x2 = [x, fliplr(x)];
        inBetween = [curve1, fliplr(curve2)];
        h=fill(x2, inBetween, 'r'); set(h, 'FaceAlpha', 0.08)
        str{1}='Training Data'; str{2}='Posterior Estimate'; 
        legend(str,'Location','Best','interpreter','Latex','FontSize',30);set(gca,'FontSize',30);
        xlim([min(data.Xtrain) max(data.Xtrain)]);
        hold off;
    else 
        figure;
        hold on; x =1:ntrain;
        plot(x,data.ytrain,'LineWidth',1.5 );
        plot(x,mu_posterior(T-ntrain+1:end),'LineWidth',1.5,'LineStyle','--'); 
        xlabel('sample index','interpreter','Latex','FontSize',30)
        ylabel('Target Value','interpreter','Latex','FontSize',30); 
        curve1 = mu_posterior(T-ntrain+1:end)+3*Sigma_posterior(T-ntrain+1:end);
        curve2 = mu_posterior(T-ntrain+1:end)-3*Sigma_posterior(T-ntrain+1:end);
        plot(x, curve1, 'r', 'LineWidth', 1);
        plot(x, curve2, 'b', 'LineWidth', 1);
        x2 = [x, fliplr(x)];
        inBetween = [curve1, fliplr(curve2)];
        h=fill(x2, inBetween, 'r'); set(h, 'FaceAlpha', 0.08)
        str{1}='Training Data'; str{2}='Posterior Estimate'; 
        legend(str,'Location','Best','interpreter','Latex','FontSize',30);set(gca,'FontSize',30);
        hold off;
    end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
figure;
plot(M(1:t),'LineWidth',2);
xlabel('$t$, number of samples','interpreter','Latex','FontSize',30)
 ylabel('Model Order','interpreter','Latex','FontSize',30)
 set(gca,'FontSize',30);
