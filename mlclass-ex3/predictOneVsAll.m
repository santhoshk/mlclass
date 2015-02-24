function p = predictOneVsAll(all_theta, X)
%PREDICT Predict the label for a trained one-vs-all classifier. The labels 
%are in the range 1..K, where K = size(all_theta, 1). 
%  p = PREDICTONEVSALL(all_theta, X) will return a vector of predictions
%  for each example in the matrix X. Note that X contains the examples in
%  rows. all_theta is a matrix where the i-th row is a trained logistic
%  regression theta vector for the i-th class. You should set p to a vector
%  of values from 1..K (e.g., p = [1; 3; 1; 2] predicts classes 1, 3, 1, 2
%  for 4 examples) 

m = size(X, 1);
num_labels = size(all_theta, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned logistic regression parameters (one-vs-all).
%               You should set p to a vector of predictions (from 1 to
%               num_labels).
%
% Hint: This code can be done all vectorized using the max function.
%       In particular, the max function can also return the index of the 
%       max element, for more information see 'help max'. If your examples 
%       are in rows, then, you can use max(A, [], 2) to obtain the max 
%       for each row.
%       

%z will be a k * m matrix (because we multiply all_theta = k*n+1 with X' = n+! * m)
%btw, each row in z will be nothing but theta' * xi, i.e exactly our hypothesis
z = all_theta * X';

%each elt in guess will have some prob
%in particular, for column 1, there will be k elements, and ith one will be the 
%prob that all_theta(i) guesses it correctly
%and hence, the max of each column is our predicted probability and the index of that
%elt is our predicted class
guess = sigmoid(z);

%im will be the vector of max indices, and m will be the max elts themselves which we do not want
[m,im] = max(guess,[],1);

%im is a row vector and p is a col vector, so just transpose to get the result classes
p = im';

% =========================================================================


end
