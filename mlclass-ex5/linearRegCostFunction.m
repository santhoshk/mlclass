function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

%
% 1. calculate the regularized linear regression cost function
%
%calculate the unregularized part of J(theta)
unreg = (X * theta) - y;
unreg = unreg .^ 2;

reg = theta(2:end)' * theta(2:end);
%ignore the first term in regularization

J = ((1/(2*m)) * sum(unreg)) + ((lambda/(2*m)) * sum(reg));


%
% 2. calculate the regularized linear regression gradient
%
Hyp_m_1 = X * theta;
err_m_1 = Hyp_m_1 - y;
grad_n_1 = X' * err_m_1;
grad_n_1 = (1/m) * grad_n_1;
grad_n_1(2:end) = grad_n_1(2:end) + (lambda/m)*theta(2:end);
grad = grad_n_1;


% =========================================================================

grad = grad(:);

end
