function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

%add a column of ones to the x vector.
a1_input = [ones(m,1) X]';
%because we are going to multiply the rows of Theta with the columns of a1, we have transposed a1

z2 = Theta1 * a1_input;
a2_output = sigmoid(z2);

% ; ==> adds a row of ones to a2_output to form a2_input
%note that [ones(c,1) A] would add a column of ones to A, assuming A has c rows.
a2_input = [ones(1,m);a2_output];
z3 = Theta2 * a2_input;
a3_output = sigmoid(z3);
[m,im] = max(a3_output,[],1);
p = im';

% =========================================================================


end
