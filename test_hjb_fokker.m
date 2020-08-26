% Test script for HJB and LQR controls for the Fokker-Planck equation
% Input (model and approximation) parameters are read from the keyboard.
% You may simply hit enter to choose the default parameters which
% correspond to the experiment in the paper.
% Output variables are stored in the main Matlab workspace. Some of the
% interesting ones are:
%   V: a cell array of TT cores of the value function approximation
%   t: time points from the test ODE solve
%   X: a trajectory from the HJB-controlled ODE
%   ux: HJB control signal
%   cost: a matrix of running [state, control] costs
%   Xlqr, ulqr, cost_lqr: same for the full LQR-controlled ODE test
%   Xlqrr, ulqrr, cost_lqrr: same for the reduced LQR-controlled ODE test
%   Xunc, cost_unc: same for the uncontrolled ODE test (uunc=0)

nx = parse_parameter('Number of discretisation points in space nx (default 1023): ', 1023);
d = parse_parameter('Reduced model dimension d (default 10): ', 10);
sigma = parse_parameter('Destabilisation shift sigma (default 0.2): ', 0.2);
gamma = parse_parameter('Control regularization parameter gamma (default 1e-2): ', 1e-2);

nv = parse_parameter('Number of Legendre polynomials for HJB nv (default 5): ',  5);
av = parse_parameter('Size of the domain [-av,av] (default 20): ', 20);
tol = parse_parameter('TT approximation and stopping tol (default 1e-4): ', 1e-4);
mu = parse_parameter('Initial shift mu (default 5): ', 5);


% Generate FP matrices
[A,B,N,Q,R,iR,rho,e,hx,xd]=fp1d(nx);
C=eye(size(A,1));

% Model reduction (bilinear balanced truncation)
[Ar,Br,Cr,Nr,Vr,Wr,Dr]=fpmr(A,B,C,N,d);


% Dyn system functions
ffun = @(x,i)(x*Ar(i,:)' + sigma*x(:,i));
gfun = @(x,i)(x*Nr(i,:)' + Br(i,:)');

% cost function for reduced model
lfun = @(x)sum((x*(sqrt(hx)*R*Q*Vr)').^2, 2);

% Anfangsbedingung
x0 = exp(-(xd'-3.8).^2*2); % difficult right-shifted initial guess
x0 = x0/(e'*x0);
x0r=Q'*iR*x0;

% LQR controllers for original and reduced systems
[K,~,e]=lqr(A+sigma*speye(nx-1),B,Q'*(R'*hx*R)*Q,gamma);
[Kr,S,~]=lqr(Ar+sigma*eye(d),Br,Vr'*Q'*(R'*hx*R)*Q*Vr,gamma);


% Solve HJB
V = hjb_leg(d, nv, av, ffun, gfun, lfun, gamma, tol, mu, @(x)quadratic_fun(x,S));


V = core2cell(V);
% HJB control function
controlfun = @(x)controlfun_leg(x*Wr,-av,av,V,gfun,gamma, inf);
% ODE function with HJB control
odefun_hjb = @(x)A*x + controlfun(x').*(N*x+B);

% Control cost function handle
ccostfun = @(x)gamma*(x.^2);
% State cost handle (full model)
lfun = @(x)sum((x*(sqrt(hx)*R*Q)').^2, 2);


% Solve HJB-controlled ODE
[t,X]=ode15s(@(t,x)odefun_hjb(x),0:0.1:20,x0r);
ux = controlfun(X);
cost = [lfun(X)  ccostfun(ux)];
% Integrate the cost using trapezoidal rule
fprintf('total HJB cost = %3.6f\n', sum((t(2:end)-t(1:end-1)).*0.5.*sum(cost(1:end-1,:)+cost(2:end,:), 2)));


% Solve full LQR-controlled ODE
[tlqr,Xlqr]=ode15s(@(t,x)(A*x+(-K*x)*N*x+B*(-K*x)),0:0.1:20,x0r);
% Control and cost
ulqr = -Xlqr*K';
cost_lqr = [lfun(Xlqr)  ccostfun(ulqr)];
fprintf('full LQR cost = %3.6f\n', sum((tlqr(2:end)-tlqr(1:end-1)).*0.5.*sum(cost_lqr(1:end-1,:)+cost_lqr(2:end,:), 2)));

% Solve reduced LQR-controlled ODE
[tlqr,Xlqrr]=ode15s(@(t,x)A*x+(-Kr*Wr'*x)*N*x+B*(-Kr*Wr'*x),0:0.1:20,x0r);
% Control and cost
ulqrr = -Xlqrr*Wr*Kr';
cost_lqrr = [lfun(Xlqrr)  ccostfun(ulqrr)];
fprintf('reduced LQR cost = %3.6f\n', sum((tlqr(2:end)-tlqr(1:end-1)).*0.5.*sum(cost_lqrr(1:end-1,:)+cost_lqrr(2:end,:), 2)));


% Uncontrolled solution
[tunc,Xunc]=ode15s(@(t,x)(A*x),0:0.1:20,x0r);
cost_unc = lfun(Xunc);
fprintf('total UNC cost = %3.6f\n', sum((tunc(2:end)-tunc(1:end-1)).*0.5.*sum(cost_unc(1:end-1,:)+cost_unc(2:end,:), 2)));

