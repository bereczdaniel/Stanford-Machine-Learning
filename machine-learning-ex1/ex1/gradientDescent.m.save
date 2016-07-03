function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters
    sumForThetaOne = 0;
    sumForThetaTwo = 0;

    sumForThetaOne = sum(theta(1) * X(:,1)) + sum(theta(2) * X(:,2)) - sum(y);
    sumForThetaTwo = sum(theta(1) * X(:,2)) + sum(theta(2) * (X(:,2) .^2)) - sum(y .* X(:,2));

    theta(1) = theta(1) - alpha * (1 / m) * sumForThetaOne;
    theta(2) = theta(2) - alpha * (1 / m) * sumForThetaTwo;
    
    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end;

end;
