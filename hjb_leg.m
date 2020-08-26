function [V] = hjb_leg(d, nv, av, ffun, gfun, lfun, gamma, tol, mu, V0fun, umax)
% Solves the HJB equation with unconstrained or weakly constrained control,
% (f + g u)*DV + gamma W(u) + \ell(x) = 0,
% where W(u) can be u^2 or \int 2 umax * tanh^{-1}(u/umax) du,  u \in R.
%   function [V] = HJB_LEG(d, nv, av, ffun, gfun, lfun, gamma, tol, mu, V0fun, umax)
% Uses the Gauss-Newton iteration and spectral Galerkin scheme, see Alg 3.2
%
% Inputs:
%   d: number of state variables x
%   nv: number of Legendre polynomials introduced in each variable x_k
%   av: size of the domain x \in [-av, av]^d
%   ffun: a @(X,i) handle of a function computing the component f_i of the
%         dynamics force vector f(x). Here, X is a M x d matrix of M 
%         positions of x where the function should be computed and returned
%         as a vector of length M.
%   gfun: a @(X,i) handle of a function computing the component g_i of the
%         actuator vector g(x). The interface is similar to ffun.
%   lfun: a @(X) handle of a function computing the running cost \ell(x).
%   gamma: control regularization parameter (>0)
%   tol: truncation and stopping tolerance for TT algorithms (0<tol<1)
%   mu: initial shift parameter in the linear system (A+mu I) V = b+mu V_p
%
% Further inputs are optional and can be omitted or empty:
%   V0fun: a @(X) handle of a function computing the initial guess of the
%          value function V(x). A zero function is used by default.
%   umax: the size of the control constraint interval [-umax, umax].
%         Unconstrained control (umax=inf) is used by default.
%
% Output:
%   V: a tt_tensor of expansion coefficients of the approximated value
%      function.
%

% Number of basis functions for V
nv = max(nv,3);
% Number of quadrature points for element computation
nq = nv*2;
% Generate Gauss points and weights
[x1v,w1v] = lgwt(nq, -av, av);

% Legendre polynomials at x1v
[P,dPdx] = legendre_rec(x1v, -av, av, nv-1);

% Expand grids to all variables
X = cell(d,1);
for i=1:d
    X{i} = tt_tensor(x1v);
    if (i>1)
        X{i} = tkron(tt_ones(nq, i-1), X{i});
    end
    if (i<d)
        X{i} = tkron(X{i}, tt_ones(nq, d-i));
    end
end

% We will need the identity matrix
I = tt_eye(nv, d);
% and constant vector in the basis
e = {P'*w1v};
% Test polynomials with quadrature weights - for computing coefficients
PWd = tt_matrix(repmat({diag(w1v)*P},d,1));

% Initial Guess
V = tt_zeros(nv,d);
if (nargin>9)&&(~isempty(V0fun))    
    V = approximate_fun(V0fun, X, tol, 'V0'); % on quadrature nodes
    V = PWd'*V; % coefficients in the basis
end

% Eliminate the constant from V (it's in the kernel of HJB op)
e{1} = e{1}/norm(e{1});
e = tt_tensor(repmat(e,1,d));
V = V - dot(e,V)*e;

% Truncation tolerance of the operator (coefficients)
tol_op = max(tol*1e-4, 1e-14); 
% It must be much smaller than the solution tolerance to represent the
% operator spectrum well

% Prepare TT tensors of functions and operator parts f*grad, g*grad
fGradq = cell(1,d);
gGradq = cell(1,d);
for i=1:d
    % Expansion coeffs of functions
    f = approximate_fun(@(x)ffun(x,i), X, tol_op, sprintf('f_%d',i));
    g = approximate_fun(@(x)gfun(x,i), X, tol_op, sprintf('g_%d',i));
    % Operator part starts with gradient
    G = tt_matrix(dPdx);
    if (i>1)
        G = tkron(tt_matrix(repmat({P},i-1,1)), G);
    end
    if (i<d)
        G = tkron(G, tt_matrix(repmat({P},d-i,1)));
    end
    % In the space of quadrature nodes, coefficients are just diag matrices
    fGradq{i} = tt_tensor(diag(-f)*G);
    gGradq{i} = tt_tensor(diag(g)*G);
end
% Parts of the operators were cast to tt_tensors to sum them up efficiently
fGradq = amen_sum(fGradq, ones(d,1), tol_op, 'kickrank', 16);
% Convert the whole f*grad back to tt_matrix
fGradq = tt_matrix(fGradq, nq*ones(d,1), nv*ones(d,1));
fGradq = PWd'*fGradq;
gGradq = amen_sum(gGradq, ones(d,1), tol_op, 'kickrank', 16);
% The row space is still in the quadrature space, since we need to add the
% nonlinear term later on
gGradq = tt_matrix(gGradq, nq*ones(d,1), nv*ones(d,1));

% Running cost coefficients
ell = approximate_fun(lfun, X, tol_op, '\ell');
ell = PWd'*ell;

tol_u = tol; % truncation tolerance for the control

% Check for default umax
if (nargin<11)||(isempty(umax))
    umax = inf;
end

%%% Initialise the Gauss-Newton policy iteration
% Control tensor on the quadrature grid
u = amen_mm(gGradq, V/(-2*gamma), tol);
if (isinf(umax))
    uc = u; % unconstrained control
else
    uc = amen_cross_s({u}, @(x)(umax*(1-tol))*tanh(x/(umax*(1-tol))), tol_u);  % Constrained control
    % Note that TT errors can overshoot by tol, compensate for that by a
    % tighter constraint
end

err_prev = inf;
gv2 = uc;
Errors = zeros(0,1);

tol_stop = tol; % Stopping tolerance of the policy iteration
tol = 0.1; % First iterations are inaccurate anyway, speed them up

% Policy iteration runs here
for iter=1:1000  % shouldn't be reached normally
    Vprev = V;
        
    % Control coefficients
    Pu = PWd'*diag(-uc);
    
    % Jacobian
    J = fGradq + Pu*gGradq;
    J = round(J, tol_op);
    
    fprintf('running cross for L(u)\n');
    % Control part of the cost for the right hand side and residual
    if (isinf(umax))
        % unconstrained control
        gv2 = amen_mm(diag(uc), gamma*uc, tol_u, 'x0', gv2);
    else
        % constrained control
        gv2 = amen_cross_s({uc}, @(x)(2*gamma) * (umax.*x.*atanh(x/umax) + umax.^2*log(1-x.^2/umax.^2)/2), tol_u, 'y0', gv2);
    end
    rhs = ell + PWd'*gv2;
    rhs = round(rhs, tol_u);
    
    % Check residual convergence
    resid_nonlin = norm(fGradq*V + PWd'*gv2 - ell)/norm(ell);
    
    % Solve the linearised shifted system
    [V,~,swp] = amen_solve(J+mu*I, rhs+mu*V, tol, struct('max_full_size', 0, 'nswp', 7, 'trunc_norm', 'resid', 'local_iters', 100, 'resid_damp', 2), V);
    % Eliminate the kernel (constant)
    V = V - dot(e, V)*e;
    % Control tensor on the quadrature grid
    u = amen_mm(gGradq, V/(-2*gamma), tol_u, 'x0', u);
    if (isinf(umax))
        uc = u; % unconstrained control
    else
        uc = amen_cross_s({u}, @(x)(umax*(1-tol))*tanh(x/(umax*(1-tol))), tol_u, 'y0', uc);  % Constrained control
    end
    
    % Check increment convergence
    err = norm(V-Vprev)/norm(V);
    fprintf('\niter=%d, err=%3.3e, mu = %3.3e, resid(nl)=%3.3e, err/err_prev=%g\n', iter, err, mu, resid_nonlin, err/err_prev);
    Errors(iter) = err;
    semilogy(1:iter, Errors, '.-');
    title('Increment |V-V_{prev}|/|V|');
    xlabel('iteration');
    drawnow;
    
    if (err<tol_stop)
        break;
    end
    
    % Update the shift
    beta = 1.02;
    if ((swp>3)&&(iter>1))
        mu = mu*2;
    elseif (err<err_prev)
        mu = mu/beta;
    end
    % And the truncation tolerance
    err_prev = err;
    tol = min(err*0.3, 0.1);
    tol_u = tol;
end
end


% A helper function to safeguard TT-Cross by several runs if necessary
function [F]=approximate_fun(fun, X, tol, fname)
err_tt = inf;
while (err_tt>tol*2)
    F = amen_cross_s(X, fun, 0, 'kickrank', 16, 'tol_exit', tol, 'y0', 2);
    test_tt = amen_cross_s(X, fun, 0, 'kickrank', 16, 'tol_exit', tol, 'y0', 2);
    err_tt = norm(F-test_tt)/norm(F);
    fprintf('TT-Cross error in %s = %3.3e\n', fname, err_tt);
end
F = round(F, tol);
end
