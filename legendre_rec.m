% Normalized Legendre polynomials and derivatives (recursive formula) on [a,b]
%   function [P,dPdx] = LEGENDRE_REC(x,a,b,n)
% Inputs:
%   x: a vector of points where the polynomials should be evaluated
%   a: left boundary of the interval
%   b: right boundary of the interval
%   n: largest degree
%
% Outputs:
%   P: a numel(x) x (n+1) matrix of polynomial values
%   dPdx: a numel(x) x (n+1) matrix of derivative values
%
function [P,dPdx] = legendre_rec(x,a,b,n)
M = numel(x);
x = (x(:)-a)*2/(b-a)-1;
P = ones(M, n+1);
dPdx = zeros(M, n+1);
P(:,2) = x;
dPdx(:,2) = ones(M, 1);
for i=1:n-1
    P(:,i+2) = (2*i+1)/(i+1)*x.*P(:,i+1) - i/(i+1)*P(:,i);
    dPdx(:,i+2) = (2*i+1)/(i+1)*(x.*dPdx(:,i+1) + P(:,i+1)) - i/(i+1)*dPdx(:,i);
end
P = P*diag(sqrt((0:n)+0.5)/sqrt((b-a)/2));
dPdx = dPdx*diag(sqrt((0:n)+0.5)/sqrt((b-a)/2))/((b-a)/2);
end
