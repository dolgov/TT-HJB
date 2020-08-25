% System matrices and right hand sides of the Fokker-Planck equation with
% bilinear control
%   function [tA,tB,tN,Q,R,iR,rho,e,hx,xd]=FP1D(nx)
% 
function [tA,tB,tN,Q,R,iR,rho,e,hx,xd]=fp1d(nx)
xa = -6;
xb = 6;
nu = 1;
hx = (xb-xa)/(nx+1);
xd = xa+hx:hx:xb-hx;
 

% backward Kolmogorov equation
% d/dt y = nu*Delta*y - < nabla y , nabla W >
% < nabla y , normal > = 0
%
W = @(x) ( ((0.5*x.^2 - 15).*x.^2 + 119).*x.^2 + 28*x + 50) / 200;
dWdx  = @(x) (3*x.^5)/200 - (3*x.^3)/10 + (119*x)/100 + 7/50;

ex = ones(nx,1);
I = speye(nx);
Laplace_x = 1/hx^2*spdiags([ex -2*ex ex], -1:1, nx, nx);
Grad_x_fw = 1/hx*spdiags([-ex ex],0:1,nx,nx);
Grad_x_bw = 1/hx*spdiags([-ex ex],-1:0,nx,nx);
Grad_x = 1/(2*hx)*spdiags([-ex 0*ex ex],-1:1,nx,nx);

dWx = spdiags(reshape(dWdx(xd)',nx,1),0,nx,nx);

e1x = I(:,1);
enx = I(:,nx);

O = sparse(nx,nx);


A = nu*Laplace_x- min(dWx,O)*Grad_x_fw - max(dWx,O)*Grad_x_bw;
A = A + nu/hx^2*(e1x*e1x') + nu/hx^2*(enx*enx') - 1/hx*min(dWx,0)*(enx*enx')...
    + 1/hx*max(dWx,0)*(e1x*e1x') ;

e = hx*ones(nx,1);

A = A';
n = size(A,1);% 
f = @(x,gap) (xa+gap < x && x < xa+2*gap)*1/(xb-xa)*(2/gap^2*(x-xa-gap)^3 ...
    -5/gap^3*(x-xa-gap)^3*(x-xa-2*gap)+9/gap^4*(x-xa-gap)^3*(x-xa-2*gap)^2) ...
    + (xa+2*gap <= x && x <= xb-2*gap)*1/(xb-xa)*(x-xa) ...
    + (xb-2*gap < x && x < xb -gap)*(1-2*gap/(xb-xa)+1/(xb-xa)*(x-xb+2*gap) ...
    +1/(gap^2*(xb-xa))*(x-xb+2*gap)^3-4/(gap^3*(xb-xa))*(x-xb+2*gap)^3*(x-xb+gap)...
    + 9/(gap^4*(xb-xa))*(x-xb+2*gap)^3*(x-xb+gap)^2)+ (xb-gap <= x && x<= xb);

%%
alpha = zeros(nx,1);
for i = 1:nx
    alpha(i) = f(xa+(i-1)*hx,1e-1)-0.5;
end 

%%

Diffx = Grad_x - 1/(2*hx)*(e1x*e1x') + 1/(2*hx)*(enx*enx');
% Computing control matrices N,B
Nt = -spdiags(Diffx*alpha,0,nx,nx)*Grad_x;
Nt = Nt + 1/(2*hx)*spdiags(Diffx*alpha,0,nx,nx)*(e1x*e1x') - ...
    1/(2*hx)*spdiags(Diffx*alpha,0,nx,nx)*(enx*enx');
N = Nt';

pow_err = inf;
rho = randn(n,1);
while pow_err > 1e-10 % comput rho_inf via power method
    rho_old = rho;
    rho_new = (A-speye(n))\rho;
    rho = rho_new/(e'*rho_new);
    pow_err = norm(rho-rho_old)/norm(rho_old);
end
B = N*rho;
orig_bil = @(t,x,A) A*x;





%% projection onto stable subspace
m = size(A,1)-1;
n = size(A,1);
R = [speye(m),rho(1:m);-ones(1,m),rho(n)];
iR = inv(R);
% iR = [speye(m),sparse(m,1);e(1:n)']-[rho(1:m);0]*e';
Q = [speye(m);zeros(1,m)];
C = speye(nx); 

rA = sparse(round(1e12*(iR*A*R))/1e12);
rN = sparse(round(1e12*(iR*N*R))/1e12);
rB = round(1e12*(iR*B))/1e12;
rC = C*R*Q*sqrt(hx);

tA = rA(1:m,1:m);
tN = rN(1:m,1:m);
tB = rB(1:m,:);
tC = rC(:,1:m);