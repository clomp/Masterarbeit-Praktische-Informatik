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
m = feval(meanfunc{:}, hyp.mean, xg); % initial mean
C = feval(covfunc{:}, hyp.cov, xg); % initial covariance = kernel matrix
iK = inv(C);

%%  Recursive
Nruns = 20; % number of update steps
Nx = 10; % number of observations per step
for i = 1:Nruns
    x = rand(s,Nx,1)*dx + xlim(1);
    y = fun(x) + sn*randn(s,Nx,1);

    [m, C] = recursiveGP(xg, m, C, x, y, meanfunc, covfunc, hyp, iK);
end

%% Plot
Np = 50; % number of predicted states
X = linspace(xlim(1), xlim(2), Np)';
[mp, Cp] = recursiveGP(xg, m, C, X, [], meanfunc, covfunc, hyp, iK);
[~, idx] = sort(X);
S = sqrt(diag(Cp));
f = [mp(idx) + 2*S(idx); flipdim(mp(idx)-2*S(idx), 1)];
fill([X(idx); flipdim(X(idx),1)], f, [8 7 7]/8, 'EdgeColor', [8 7 7]/8);
plot(X(idx), mp(idx), 'r-');
plot(xp, fun(xp), 'k');
plot(xg, zeros(1,Np), 'bx');
waitforbuttonpress