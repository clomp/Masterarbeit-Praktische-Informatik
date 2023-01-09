clear all; close all;

addpath(genpath('./GP'))

s = RandStream('mt19937ar','Seed', 1);
demo = 'sin'; %demo = ' ';

%% Function
if demo == 'sin'
    fun = @(x) sin(2*x);
    xlim = [0 6];
else
    fun = @(x) x/2 + 25.*x./(1+x.^2).*cos(x);
    xlim = [-10 10];
end

dx = xlim(2)-xlim(1);
xp = linspace(xlim(1), xlim(2), 50)';
hold on;
grid on; box on;
plot(xp, fun(xp), 'k');

if demo == 'sin'
    ylim([-2 2]);
else
    ylim([-10 10]);
end

%% GP
meanfunc = {@meanZero};
covfunc = {@covSEiso};
hyp.mean = [];
ell = 1/2; sf = 1; hyp.cov = log([ell; sf]);
likfunc = @likGauss; sn = 0.1; hyp.lik = log(sn);

%% Init, k = 0
if demo=='sin'; N = 20; else N = 40; end % number of basis vectors
xg = linspace(xlim(1), xlim(2), N)'; % fixed grid
m0 = feval(meanfunc{:}, hyp.mean, xg); % initial mean
C0 = feval(covfunc{:}, hyp.cov, xg); % initial covariance = kernel matrix
iK = inv(C0);
plot(xg, m0, 'b--');

waitforbuttonpress

%%  k = 1
Nx = 20; % number of observations
x1 = rand(s,Nx,1)*dx + xlim(1);
y1 = fun(x1) + sn*randn(s,Nx,1);

[m1, C1] =  recursiveGP(xg, m0, C0, x1, y1, meanfunc, covfunc, hyp, iK);
plot(xg, m1, 'r--');


%%  k = 2
Nx = 10; % number of observations
x2 = rand(s,Nx,1)*dx + xlim(1);
y2 = fun(x2) + sn*randn(s,Nx,1);
X = [xg; x2];
X = linspace(xlim(1), xlim(2), N)';

% Plot
[m, C] = recursiveGP(xg, m1, C1, X, [], meanfunc, covfunc, hyp, iK);
[~, idx] = sort(X);
S = sqrt(diag(C));
f = [m(idx) + 2*S(idx); flipdim(m(idx)-2*S(idx), 1)];
fill([X(idx); flipdim(X(idx),1)], f, [8 7 7]/8, 'EdgeColor', [8 7 7]/8);
plot(X(idx), m(idx), 'r-');
plot(xp, fun(xp), 'k');
plot(xg, zeros(1,N), 'bx');
waitforbuttonpress

% Update
[m2, C2] =  recursiveGP(xg, m1, C1, x2, y2, meanfunc, covfunc, hyp, iK);
plot(xg, m2, 'g--');

%%  k = 3
Nx = 20; % number of observations
x3 = rand(s,Nx,1)*dx + xlim(1);
y3 = fun(x3) + sn*randn(s,Nx,1);
%X = [xg; x3];
X = linspace(xlim(1), xlim(2), N)';

% Plot
[m, C] = recursiveGP(xg, m2, C2, X, [], meanfunc, covfunc, hyp, iK);
[~, idx] = sort(X);
s = sqrt(diag(C));
f = [m(idx) + 2*s(idx); flipdim(m(idx)-2*s(idx), 1)];
fill([X(idx); flipdim(X(idx),1)], f, [7 8 7]/8, 'EdgeColor', [7 8 7]/8);
plot(X(idx), m(idx), 'g-');
plot(xp, fun(xp), 'k');
plot(xg, zeros(1,N), 'bx');
waitforbuttonpress

% Update
[m3, C3] = recursiveGP(xg, m2, C2, x3, y3, meanfunc, covfunc, hyp, iK);
[m, C] = recursiveGP(xg, m3, C3, X, [], meanfunc, covfunc, hyp, iK);
% 
% plot(X, m, 'm--');
[~,idx]=sort(x3)
plot(x3(idx), m, 'm--');

%% Full GP
waitforbuttonpress
XX = [x1; x2; x3];
YY = [y1; y2; y3];
[mgp, Cgp] = gpr(hyp, @infExact, meanfunc, covfunc, likfunc, XX, YY, X);
plot(X, mgp, '-.', 'Color', [.2 .6 1]);