function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    delta = zeros(length(theta),1);
    for i = 1:m,
        pred = theta' * X(i,:)';
        diff = pred - y(i);
        partial = diff * X(i,:)';
        delta = delta + partial;
    end;
    delta = delta / m;

    %disp(sprintf('delta for iter %d =  %0.2f \n', iter, delta));

    theta = theta - (alpha * delta);    

    % ============================================================

    % Save the cost J in every iteration    
    cost = computeCost(X, y, theta);

    %disp(sprintf('cost for iter %d =  %0.2f \n', iter, cost));

    J_history(iter) = cost;

end

end
