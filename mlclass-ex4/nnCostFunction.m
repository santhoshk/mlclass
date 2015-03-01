function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


%first we add a column of 1's to the X matrix so that the bias unit setup to 1
X = [ones(m,1) X];
%note that the dimentions of X are now, rows = n+1; cols = m (where n = 400 in our example)

%part-1,2,3(3 has 5 steps) : Backpropagation implementation
% some of the below code is duplicated from step1, especially the forward prop, but ok for now; refactor later.
%note : 
%	size(Theta1) = size(Theta1_grad) = 25 x 401
%	size(Theta2) = size(Theta2_grad) = 10 x 26
for t = 1:m, %for each input example, calculate gradient and accumulate to Theta1_grad and Theta2_grad
	%Step1 
	%	Set the input layers values a(1) to the t-th training example, x(t)
	%	Perform forward pass
	a_1 = X(t,:)'; %note that the bias has already been added to X, a_1 is now, 401x1
	z_2 = Theta1 * a_1; %Theta1 = 25x401 and a_1 = 401x1; therefore, z_2 = 25x1
	a_2 = sigmoid(z_2); %z2 = 25 x 1
	a_2 = [1;a_2]; %a_2 = 26 x 1
	z_3 = Theta2 * a_2; %(10 x 26) x (26 x 1) = (10 x 1)
	a_3 = sigmoid(z_3); %a_3 = 10 x 1

	%Step2
	%	For each output unit k in layer 3, set delta(3)_k = (a(3)_k - y_k)
	%	y_k E {0,1} indicates whether the current training example belongs to class-k
	y1 = zeros(size(a_3));
	y1(y(t)) = 1;
	delta_3 = a_3 .- y1; %dims 10 x 1 all of them.


	%Step3
	%	For the hidden layer l=2, set 
	%	delta(2) = Theta(2)' * delta(3) .* sigmoidGradient(z_2)
	%	check on dims : Theta(2) = 10 x 26; Theta(2)' = 26 x 10; delta(3) = 10 x 1; so, delta(2) = 26 x 1
	delta_2 = (Theta2' * delta_3);
	delta_2 = delta_2(2:end); %now delta_2 = 25 x 1
	delta_2 = delta_2 .* sigmoidGradient(z_2);

	%Step4
	%	Accumulate the gradient using
	%	Theta2_grad = Theta2_grad + delta_3*a_2' (last term is, 10x1 x 1x26 = 10x26)
	%	Theta1_grad = Theta1_grad + delta_2*a_1' (last term is, 25x1 x 1x401 = 25x401)
	Theta2_grad = Theta2_grad + (delta_3 * a_2');
	Theta1_grad = Theta1_grad + (delta_2 * a_1');

	tmp = -(y1' * log(a_3)) - (((1-y1)') * log(1 - a_3)); %tmp is the inner summation 1..k represented as a vector of len k
	J = J+sum(tmp);

end;	

%part-1calculate cost
J = J/m;

%part-2regularized cost
tmp1 = Theta1(:,2:end) * Theta1(:,2:end)';
tmp1 = eye(length(tmp1)) .* tmp1;

tmp2 = Theta2(:,2:end) * Theta2(:,2:end)';
tmp2 = eye(length(tmp2)) .* tmp2;
J = J + ((lambda/(2*m)) * ( sum(sum(tmp1)) + sum(sum(tmp2))));

%Step5
%	obtain the unreg grad for the neural nw cost func by dividing the accumulated grads by m.
Theta1_grad = (1/m) * Theta1_grad;
Theta2_grad = (1/m) * Theta2_grad;

%regularized gradient
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + ((lambda/m) * (Theta1(:,2:end)));
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + ((lambda/m) * (Theta2(:,2:end)));

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
