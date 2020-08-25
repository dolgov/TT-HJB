% Computes control signal given a tt_tensor of value function coefficients
% in the Legendre basis
%   function [ux] = CONTROLFUN_LEG(x,a,b,V,gfun,gamma,umax)
% Uses unconstraint or tanh-constrained controls
% Inputs:
%   x: a N x d matrix of coordinate positions where u(x) should be evaluated
%   a: left boundary of the domain x \in [a,b]^d
%   b: right boudary of the domain
%   V: a tt_tensor of the value function
%   gfun: a @(X,i) handle of a function computing the component g_i of the
%         actuator vector f(x). Here, X is a M x d matrix of M positions of
%         x where the function should be computed and returned as a vector 
%         of length M.
%   gamma: control regularization parameter (>0)
%   umax: the size of the control constraint interval [-umax, umax].
%         Unconstrained control is specified with umax=inf
% 
% Output:
%   ux: A N x 1 vector of control values at x

function [ux] = controlfun_leg(x,a,b,V,gfun,gamma,umax)
d = numel(V);
nv = size(V{1},2);
nt = size(x,1);
ux = zeros(nt,1);
% Compute Legendre polynomials and their derivatives
[p,dp] = legendre_rec(x(:), a, b, nv-1);
p = reshape(p, nt, d, nv);
dp = reshape(dp, nt, d, nv);
% Loop over all points in x
for k=1:nt 
    % compute (g*grad) V
    for j=1:d
        Vxj = 1;
        for i=1:d
            Vxj = Vxj*reshape(V{i}, size(Vxj,2), []);
            Vxj = reshape(Vxj, nv, []);
            if (i==j)
                Vxj = reshape(dp(k,i,:), 1, [])*Vxj;
            else
                Vxj = reshape(p(k,i,:), 1, [])*Vxj;
            end
        end
        ux(k) = ux(k) - Vxj*gfun(x(k,:), j)/(2*gamma);
    end
    % Apply constraint if needed
    if (~isinf(umax))
        ux(k) = umax*tanh(ux(k)/umax);
    end
end
end

