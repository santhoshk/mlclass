function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta



val = 0;
derivVector = zeros(length(theta), 1);
partialSum = 0;
for i=1:m,
    %calculate J
    z = theta' * X(i,:)';
    partialSum += -(y(i) * log(sigmoid(z))) - ((1 - y(i)) * log(1 - sigmoid(z)));

    %calculate derivatives via vectorized operations
    z = theta' * X(i,:)';
    pred = sigmoid(z);
    diff = pred - y(i);
    partial = diff * X(i,:)';
    derivVector = derivVector + partial;
end;    


reg = 0;
for j = 2 : n,
	reg += theta(j) ^ 2;
end;
reg = (lambda / (2 * m)) * reg;

J = ((1/(m)) * partialSum) + (reg);
grad = (1/m) .* derivVector;

for k = 2 : n,
	grad(k) += (lambda / m) * theta(k);
end;



% =============================================================

end
