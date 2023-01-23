%gprPerf.m
%
%DESCRIPTION:
%    /\/\/\/\/\/\/\/\____________________/\/\/\/\/\/\/\ data genneration
%INPUTS:

function data = triangledatageneration(ntrain,ncv,ntest,m,a)
data={};

data.Xtrain=1:ntrain;
for t=1:ntrain
    if (t<=round(ntrain/3))
        firsttriangle(t)=m*(t-a)*sign(a-t) + m*a;
        data.ytrain(t)=m*(t-a)*sign(a-t) + m*a;
   elseif ((t>=1+round(ntrain/3)) && (t<=round(2*ntrain/3)))
        data.ytrain(t)=randn(1,1);
   elseif (t>round(2*ntrain/3))
        data.ytrain(t)=firsttriangle(t-(2*ntrain/3));
    end
    
end

plot(data.Xtrain,data.ytrain)
%check after this part as "a" needs to change 
% data.Xtest=ntrain+1:ntrain+ntest;
% for t=ntrain+1:ntrain+ntest
%     if (t<=ntrain+round(ntest/3))
%         firsttriangle(t)=m*(t-a)*sign(a-t) + m*a;
%         data.ytest(t)=m*(t-a)*sign(a-t) + m*a;
%     elseif (1+ntrain+round(ntest/3)<=t<=ntrain+round(2*ntest/3))
%         data.ytest(t)=randn(1,1);
%     else 
%         data.ytest(t)=firsttriangle(t-(ntrain+ round(2*ntest/3)));
%     end
% end
% 
% data.Xcv=ntrain+ntest+1:ntrain+ntest+ncv;
% for t=ntrain+ntest+1:ntrain+ntest+ncv
%     if (t<=ntrain+ntest+round(ncv/3))
%         firsttriangle(t)=m*(t-a)*sign(a-t) + m*a;
%         data.ycv(t)=m*(t-a)*sign(a-t) + m*a;
%     elseif (1+ntrain+ntest+round(ncv/3)<=t<=ntrain+ntest+round(2*ncv/3))
%         data.ycv(t)=randn(1,1);
%     else 
%         data.ycv(t)=firsttriangle(t-(ntrain+ntest+round(2*ncv/3)));
%     end
% end
 

end