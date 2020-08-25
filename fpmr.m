% Bilinear Balanced Truncation algorithm for Fokker-Planck dynamics
%   function [Ar,Br,Cr,Nr,Vr,Wr,Dr]=FPMR(A,B,C,N,r)
% Inputs: matrices A,B,C,N of the full equations
%  dx/dt = Ax + (Nx + B)u, y = Cx,
% the target rank r
% Outputs: reduced matrices, subspace bases such that Wr'*Vr = I 
%
function [Ar,Br,Cr,Nr,Vr,Wr,Dr]=fpmr(A,B,C,N,r)

%% solve X%%
resx=@(X) norm(A*X+X*A'+N*X*N'+B*B')/norm(B*B');
Xm=0*A;
err=1;
tol=1e-8;
it=1;
while(err>tol)
Xp=lyap(A,N*Xm*N'+B*B');
err=resx(Xp)
Xm=Xp;
it=it+1
end
Xl=Xm;



%% solve Y%%
resy=@(Y) norm(A'*Y+Y*A+N'*Y*N+C'*C)/norm(C'*C);
Ym=0*A;
err=1;
it=1;
while(err>tol)
Yp=lyap(A',N'*Ym*N+C'*C);
err=resy(Yp)
Ym=Yp;
it=it+1
end
Yl=Ym;

%% balancing and truncation

Xc=chol(Xl+1e-12*C);
Yc=chol(Yl);
[U,S,V]=svd(Xc*Yc');
Dr=diag(S(1:r,1:r).^(-0.5));
Vr=Xc'*U(:,1:r)*diag(Dr);
Wr=Yc'*V(:,1:r)*diag(Dr);
Ar=Wr'*A*Vr;
Nr=Wr'*N*Vr;
Br=Wr'*B;
Cr=C*Vr;