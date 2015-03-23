function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions


    %cvPredictions is a vector of cross validation predictions of size equal to cv set
    %The ith element in this vector will be 1 if the algorithm considers x-i to be anomaly
    %It will be 0, otherwise.
    cvPredictions = (pval < epsilon);

    %true positives (we predicted 1 and ground truth is also 1)
    tp = sum((cvPredictions == 1) & (yval == 1));

    %false pos (we predicted 1, but ground truth is 0)
    fp = sum((cvPredictions == 1) & (yval == 0));

    %false neg (we predicted 0, but ground truth is 1)
    fn = sum((cvPredictions == 0) & (yval == 1));

    %precision
    prec = tp / (tp + fp);

    %recall
    rec = tp / (tp + fn);

    F1 = (2 * prec * rec) / (prec + rec);

    % =============================================================

    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
