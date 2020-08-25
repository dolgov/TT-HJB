% This file is for Gauss-Newton iteration, for resolvable control u ~ g Grad V
% Uses spectral differentiation in parameters
function test_hjb_allencahn1(V,t)

d = 14;

nv = 5; % number of grid points in V
av = 3; % size of V domain, [-av, av]^d

tol = 1e-3;
gamma = 0.1;

mu = 100;
umax = inf;

% 1D Laplace
% 3 - Three-stable, F(x) = sigma*L*x + X*(1-X^2)
sigma = 0.2; % Diffusion coeff

% Characteristic function of controlled domain [omega_left, omega_right]
omega_left = -0.5;
omega_right = 0.2;

% Time horizon for testing the controls
T = 30;



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


% Prepare handles for ODE funs
ffun = @(x,i)odefun(x,Ax,i); % Dynamics
gfun = @(x,i)gxi(i)*ones(size(x,1),1); % Actuator (here just const vector)


%%%%%%%%%%%%%%%%%%% Solve HJB
if (nargin<1)
    V = [];
end
V = hjb_leg(size(Ax,1), nv, av, ffun, gfun, @(x)lfun(x,wxi), gamma, tol, mu, V, umax);


% HJB control fun
V = core2cell(V); % Disintegrate tt_tensor class to speed up many evals.
controlfun = @(x)controlfun_leg(x,-av,av,V,gfun,gamma,umax);


% Test the control by solving ODEs
% Initial state in space
x0 = 2+cos(2*pi*xi).*cos(pi*xi); 
        
% Set up integrator options
if (nargin<2)
    t = [0, T];
end
opts = odeset('RelTol', 1e-6, 'AbsTol', 1e-20);

% Create a handle for the control cost
if (isinf(umax))
    ccostfun = @(u)gamma*(u.^2);
else
    ccostfun = @(u)(2*gamma) * (umax.*u.*atanh(min(max(u/umax,-1),1)) + umax.^2*log(max(1-u.^2/umax.^2,0))/2);
end

% Solve HJB-controlled ODE
[t,x] = ode15s(@(t,x)odefun(x', Ax).' + controlfun(x').', t, x0', opts);
ux = controlfun(x); % Save the control separately
cost = sum((x.^2).*(wxi'), 2) + ccostfun(ux);
% Compute the total cost using the trapezoidal rule
fprintf('total HJB cost = %3.6f\n', sum((t(2:end)-t(1:end-1)).*0.5.*(cost(1:end-1)+cost(2:end))));

% Solve uncontrolled ODE
[tunc, xunc] = ode15s(@(t,x)odefun(x', Ax).', t, x0', opts);
cost_unc = sum((xunc.^2).*(wxi'), 2);
fprintf('total UNC cost = %3.6f\n', sum((tunc(2:end)-tunc(1:end-1)).*0.5.*(cost_unc(1:end-1)+cost_unc(2:end))));

% Solve with LQR control
[tlqr,xlqr] = ode15s(@(t,x)odefun(x', Ax).' - gxi*(K*x), t, x0', opts);
ulqr = -xlqr*K';
cost_lqr = sum((xlqr.^2).*(wxi'), 2) + gamma*ulqr.^2;
fprintf('total LQR cost = %3.6f\n', sum((tlqr(2:end)-tlqr(1:end-1)).*0.5.*(cost_lqr(1:end-1)+cost_lqr(2:end))));

% Copy vars to main space
vars = whos;
for i=1:numel(vars)
    if (exist(vars(i).name, 'var'))
        assignin('base', vars(i).name, eval(vars(i).name));
    end
end
end



% Cost fun
function [f]=lfun(x,wxi)
f = sum((x.^2).*(wxi'), 2);
end

% Dyn system functions
function [f] = odefun(x, Ax, i)
if (nargin>2)
    % Evaluate a single component f_i
    f = x*Ax(i,:).';
    x1 = x(:,i);
else
    % Evaluate the whole vector f
    f = x*Ax.';
    x1 = x;    
end

f = f + x1.*(1-x1.^2);
end
