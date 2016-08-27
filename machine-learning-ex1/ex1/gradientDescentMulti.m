function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
n = size(X, 2);
temp = zeros(n, 1);

for i = 1:num_iters


  for j = 1:n
    temp(j) = theta(j) - alpha / m * ((theta' * X' * X(:, j)) - sum(y .* X(:, j)));
  end;

  theta = temp;
  J_history(i) = computeCostMulti(X, y, theta);
    
end;

end
