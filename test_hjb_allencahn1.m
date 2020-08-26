% Test script for HJB and LQR controllers for 1D Allen-Cahn equation
%   dx/dt = sigma*L*x + x*(1-x^2) + \chi_{\omega}*u
% Input (model and approximation) parameters are read from the keyboard.
% You may simply hit enter to choose the default parameters which
% correspond to the experiment in the paper.
% Output variables are stored in the main Matlab workspace. Some of the
% interesting ones are:
%   V: a cell array of TT cores of the value function approximation
%   t: time points from the test ODE solve
%   X: a trajectory from the HJB-controlled ODE
%   ux: HJB control signal
%   cost: a vector of running costs
%   Xlqr, ulqr, cost_lqr: same for the full LQR-controlled ODE test
%   Xunc, cost_unc: same for the uncontrolled ODE test (uunc=0)

d = parse_parameter('Number of variables d (default 14): ', 14);
sigma = parse_parameter('Diffusion coefficient sigma (default 0.2): ', 0.2);
gamma = parse_parameter('Control regularization parameter gamma (default 1e-1): ', 1e-1);
umax = parse_parameter('Control constraint umax (default inf): ', inf);

% Characteristic function of controlled domain [omega_left, omega_right]
omega_left = parse_parameter('Left boundary of the controlled domain omega_left (default -0.5): ', -0.5);
omega_right = parse_parameter('Right boundary of the controlled domain omega_right (default 0.2): ', 0.2);

nv = parse_parameter('Number of Legendre polynomials for HJB nv (default 5): ',  5);
av = parse_parameter('Size of the domain [-av,av] (default 3): ', 3);
tol = parse_parameter('TT approximation and stopping tol (default 1e-3): ', 1e-3);
mu = parse_parameter('Initial shift mu (default 100): ', 100);

T = parse_parameter('Time horizon for testing the controls T (default 30): ', 30);




% Space differentiation matrices
[Gx,xi] = spectral_chebdiff(d+1,-1,1);
% Neumann BC eliminator (solve u'(-1)=u'(1)=0)
s_bound = -[Gx(1,1), Gx(1,d+2); Gx(d+2,1), Gx(d+2,d+2)] \ Gx([1,d+2], 2:(d+1));
bound = [1; d+2];
% Second order derivative
Ax = Gx*Gx;
Ax = Ax(2:(d+1), 2:(d+1)) + Ax(2:(d+1), bound)*s_bound;
Ax = Ax*sigma;
% Mass vector
G0 = Gx(1:d+1,1:d+1)';
wxi = G0\eye(d+1,1);
wxi = wxi(2:(d+1)) + wxi(1)*s_bound(1,:)';

% Boost up approx order by 1 by differentiating an exact integral
gxi = double(xi>=omega_left & xi<=omega_right).*(xi-omega_left) + double(xi>omega_right).*(omega_right-omega_left);
gxi = Gx*gxi;
gxi = gxi(2:d+1);

% Remove boundary points
xi = xi(2:(d+1));

% Exact solution of the linear part
Ax_ric = Ax + eye(d); % linearised model at origin
[K,Pi,e] = lqr(Ax_ric,gxi,diag(wxi),gamma);


% Prepare handles for ODE component functions for HJB
ffun = @(x,i)x*Ax(i,:).' + x(:,i).*(1-x(:,i).^2); % Dynamics
gfun = @(x,i)gxi(i)*ones(size(x,1),1); % Actuator (here just const vector)

% State cost function
lfun = @(x)sum((x.^2).*(wxi'), 2);


%%%%%%%%%%%%%%%%%%% Solve HJB
V = hjb_leg(size(Ax,1), nv, av, ffun, gfun, lfun, gamma, tol, mu, [], umax);


% HJB control fun
V = core2cell(V); % Disintegrate tt_tensor class to speed up evaluations
controlfun = @(x)controlfun_leg(x,-av,av,V,gfun,gamma,umax);
% ODE right-hand side function (full vector)
odefun = @(x)x*Ax.' + x.*(1-x.^2);

% Test the control by solving ODEs
% Initial state in space
x0 = 2+cos(2*pi*xi).*cos(pi*xi); 
        
% Set up integrator options
t = [0, T];
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-20);

% Create a handle for the control cost
if (isinf(umax))
    ccostfun = @(u)gamma*(u.^2);
else
    ccostfun = @(u)(2*gamma) * (umax.*u.*atanh(min(max(u/umax,-1),1)) + umax.^2*log(max(1-u.^2/umax.^2,0))/2);
end

% Solve HJB-controlled ODE
[t,X] = ode15s(@(t,x)odefun(x').' + controlfun(x').', t, x0', opts);
ux = controlfun(X); % Save the control separately
cost = sum((X.^2).*(wxi'), 2) + ccostfun(ux);
% Compute the total cost using the trapezoidal rule
fprintf('total HJB cost = %3.6f\n', sum((t(2:end)-t(1:end-1)).*0.5.*(cost(1:end-1)+cost(2:end))));

% Solve uncontrolled ODE
[tunc, Xunc] = ode15s(@(t,x)odefun(x').', t, x0', opts);
cost_unc = sum((Xunc.^2).*(wxi'), 2);
fprintf('total UNC cost = %3.6f\n', sum((tunc(2:end)-tunc(1:end-1)).*0.5.*(cost_unc(1:end-1)+cost_unc(2:end))));

% Solve with LQR control
[tlqr,Xlqr] = ode15s(@(t,x)odefun(x').' - gxi*(K*x), t, x0', opts);
ulqr = -Xlqr*K';
cost_lqr = sum((Xlqr.^2).*(wxi'), 2) + gamma*ulqr.^2;
fprintf('total LQR cost = %3.6f\n', sum((tlqr(2:end)-tlqr(1:end-1)).*0.5.*(cost_lqr(1:end-1)+cost_lqr(2:end))));

