function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

%test_c = [0.01 0.03 0.1 0.3 1 3 10 30];
%test_sigma = [0.01 0.03 0.1 0.3 1 3 10 30];
%res = zeros(64,3);
%i = 0;
%
%for this_c = test_c,
%	for this_sigma = test_sigma,
%		model = svmTrain(X, y, this_c, @(x1, x2) gaussianKernel(x1, x2, this_sigma));
%		pred = svmPredict(model, Xval);
%		pred_error = mean(double(pred ~= yval));
%
%		i++
%		res(i,:) = [this_c; this_sigma; pred_error];
%		res(i,:)
%	end
%end	
%
%res
%res(:,3)
%[m,im] = min(res(:, 3))
%C = res(im,1);
%sigma = res(im,2);
%C
%sigma

% If we run the above code, we get C = 1, sigma = .1; simply plugging in the value here
C = 1;
sigma = 0.1;

% =========================================================================

end
