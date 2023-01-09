%grad_search_dictionary_y
%
%DESCRIPTION:
%    To compensate for lossy compression, this code searches for optimal dictionary
%    and y using alternating gradient descent
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
%    *D
function [Dcap,y]=grad_search_dictionary_y(D,Daug,y,yaug,kernel,x,noise_prior,c,eta,t)
% in the simulation vector y is taken as a row vector
Dcap=D;
%eta=0.01;

KDtDt=kernel.f(Daug,Daug);
k_XX=kernel.f(x,x);
k_DtX=kernel.f(Daug,x);

%compute posterior distribution parameters for original dictionary D_tilde
    mu_Dt=k_DtX'/(KDtDt + noise_prior^2*eye(size(KDtDt)))*yaug';
    Sigma_Dt=k_XX-k_DtX'/(KDtDt + noise_prior^2*eye(size(KDtDt)))*k_DtX;
    
KDcapDcap=kernel.f(Dcap,Dcap);
k_XX=kernel.f(x,x);
k_DcapX=kernel.f(Dcap,x);

M=size(Dcap,2);%no. of dictionary elements
    p=size(x,1);%size of the vector (individual dictionary element)
    
    %compute current posterior distribution parameters for dictionary Dcap
    %This is used in the first iteration
    mu_Dcap=k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*y';
    Sigma_Dcap=k_XX-k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*k_DcapX ;
    %%%%print the posterior distribution parameters
%     if(t>178)
%     fprintf('original mu: %d and sigma:%d and intial Dcap parameters mu:%d and Sigma:%d\n',mu_Dt,Sigma_Dt,mu_Dcap,Sigma_Dcap);
%     end
for itr=1:20
       
    for i=1:M
        for k=1:p
            %ith dictionary element
            di=Dcap(:,i);
            derv_k_DcapX_dik=zeros(M,1);
            derv_KDcapDcap_dik=zeros(M,M);
            
            derv_k_DcapX_dik(i)=(-1/c^2)*kernel.f(di,x)*(di(k)-x(k));
                      
            l=1:M;
            derv_KDcapDcap_dik(i,:)=(-1/c^2).*kernel.f(di,Dcap(:,l)).*(di(k)-Dcap(k,l));
         %   derv_KDcapDcap_dik(:,i)=(1/c^2).*kernel.f(Dcap(:,l),di).*(Dcap(k,l)-di(k));
            derv_KDcapDcap_dik(:,i)=(-1/c^2).*kernel.f(di,Dcap(:,l)).*(di(k)-Dcap(k,l));
            
            
            derv_mu_Dcap_dik=derv_k_DcapX_dik'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*y' ...
                -k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*derv_KDcapDcap_dik/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*y';
            
            derv_sigma_Dcap_dik=-derv_k_DcapX_dik'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*k_DcapX ...
                + k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*derv_KDcapDcap_dik/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*k_DcapX ...
                - k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*derv_k_DcapX_dik ;
            
            derv_mu_Dcap(k,i)=derv_mu_Dcap_dik;
            derv_sigma_Dcap(k,i)=derv_sigma_Dcap_dik;
           
        end
    end
    
    derv_dHcap_Dcap=2*(mu_Dcap-mu_Dt)*derv_mu_Dcap + 2*(Sigma_Dcap-Sigma_Dt)*derv_sigma_Dcap;
    %gradient descent step
    Dcap=Dcap-eta*derv_dHcap_Dcap;
    
    %update posterior parameters wrt Dcap for y
    KDcapDcap=kernel.f(Dcap,Dcap);
    k_XX=kernel.f(x,x);
    k_DcapX=kernel.f(Dcap,x);
    
    %compute updated posterior distribution parameters with updated
    %dictionary Dcap
    mu_Dcap=k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*y';
    Sigma_Dcap=k_XX-k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*k_DcapX ;
    
    temp =k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)));
    
    derv_mu_Dcap_y=temp;
    derv_dHcap_y=2*(mu_Dcap-mu_Dt)*derv_mu_Dcap_y;
    y=y-eta*derv_dHcap_y;
    %Update again posterior distribution parameters as y got updated so
    %that in the next iteration when D is calculated it will use updated
    %posterior distribution parameters wrt y
    %Sigma need'nt be calulated again but for sake of completeness we again
    %wrote it
    mu_Dcap=k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*y';
    Sigma_Dcap=k_XX-k_DcapX'/(KDcapDcap + noise_prior^2*eye(size(KDcapDcap)))*k_DcapX ;
    %%%%print the posterior distribution parameters
%     if(t>178)
%     fprintf('original mu: %d and sigma:%d and updated Dcap parameters mu:%d and Sigma:%d\n',mu_Dt,Sigma_Dt,mu_Dcap,Sigma_Dcap);
%     end
end
    
    
    
    
    
    
                
            
            


