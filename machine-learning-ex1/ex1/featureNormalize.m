function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
X_norm = X;
mu = zeros(1, size(X, 2));
sigma = zeros(1, size(X, 2));

mean_x1 = mean(X(:,1))
mean_x2 = mean(X(:,2))

std_x1 = std(X(:,1))
std_x2 = std(X(:,2))

X_norm(:,1) = (X_norm(:,1) - mean_x1) ./ std_x1;
X_norm(:,2) = (X_norm(:,2) - mean_x2) ./ std_x2;

mu = X_norm(:,1);
sigma = X_norm(:,2);
[X_norm, mu, sigma];





% ============================================================

end
