% Computes the quadratic polynomial V(x) = x*S*x' given the matrix S and a
% row vector x. It can also evaluate a vector of values of V if x is given
% as a matrix, computing V(x_i) row by row.
function [V] = quadratic_fun(x,S)
V = zeros(size(x,1),1);
for i=1:size(x,1)
    V(i) = x(i,:)*S*x(i,:)';
end
end
