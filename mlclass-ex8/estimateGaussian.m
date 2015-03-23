function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%

%column wise of sum of X (===sum(X,0)) will give a row 
%the ith elt in that row will be the sum of i-th feature in all m examples
mu = (1/m)*sum(X,0)';

%transpose of X will put the examples (1..m), one in each column
%then subtract mean(i) from i-th row of every example
%then square; then sum row wise

%to subtract from X', lets repeat the mean, m number of times
rep_mu = repmat(mu,1,m); 
sigma2 = (1/m) * (sum(((X' - rep_mu) .^ 2), 2));


% =============================================================


end
