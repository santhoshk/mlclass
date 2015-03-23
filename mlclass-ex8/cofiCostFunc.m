function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

vec = ((((X * Theta') - Y) .* R) .^ 2);
J = sum(sum(vec)) * (1/2);


for i=1:num_movies,
	%list of all the users who have rated movie i
	idx = find(R(i,:) == 1);

	%Theta_temp is the parameter matrix for all users who rated movie i; rated_users * num_features
	Theta_temp = Theta(idx,:);
	Y_temp = Y(i,idx);

	%in Theta_temp', the users will be in the columns; so it will be num_features * rated_users
	%X(i,:) * Theta_temp' will multiply 1 movie row features (i.e 1*num_features) with all users who rated that movie (num_features * rated_users)
	%so we get (1*rated_users) output. 
	%then subtract Y_temp to get first part of gradient. This will be (1*rated_users) size. This first multiplier is common for all n partial derivatives in X_grad
	%The 2nd multiplier is Theta_j for all j that have rated the movie. This is nothing by Theta_temp
	%multiply first multiplier by Theta_temp, we get (1*rated_users) * (rated_users*num_features) = 1*num_features output. This is exactly X_grad(i,:)
	X_grad(i,:) = (((X(i,:) * Theta_temp') - Y_temp) * (Theta_temp));	

	%adding regularization to the gradient
	X_grad(i,:) += lambda * X(i,:);
end;


%we will use similar logic for calculating Theta_grad for each theta_j
for j=1:num_users,
	%all movies rated by user j (num_ratings * 1)
	idx1 = find(R(:,j) == 1); 

	%X_temp1 is the parameter vector for all movies rated by user j (num_ratings * num_features)
	X_temp1 = X(idx1,:);

	%the ratings of all movies rated by user j (num_ratings * 1)
	Y_temp1 = Y(idx1,j);

	Theta_grad(j,:) =  (((X_temp1 * Theta(j,:)') - Y_temp1)' *  (X_temp1))';

	%adding regularization to the gradient
	Theta_grad(j,:) += lambda * Theta(j,:);
end;


%adding regularization to the cost func.
reg_Theta = sum(sum((Theta .^ 2))) * (lambda/2);
reg_X = sum(sum(X .^ 2)) * (lambda/2);
J = J + reg_Theta + reg_X;








% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
