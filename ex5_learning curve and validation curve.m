%%%===================Exercise 5: Learning curve and validation curve===================
load ('/Users/jianfeng/learning/Machine learning/machine-learning-ex5/ex5/ex5data1.mat');
plot(X,y,'rx','MarkerSize',10,'LineWidth',1.5);

% Regularized linear regression cost function
function [J,grad] = linearRegCostFunction(theta, X, y, lambda)
    m = size(X,1)
    X = [ones(m,1) X]
    J = sum((X * theta - y).^2)/2/m + lambda * sum(theta(2:end).^2)/2/m
    grad = X' * (X * theta - y)/m + lambda * theta ./ m 
    grad(1) = X(:,1)' * (X * theta - y)/m
end

Theta = [1;1];
lambda = 1;
[J, grad] = linearRegCostFunction(Theta, X, y, lambda);

lambda = 0;
options = optimset('GradObj','On','MaxIter', 200);
[theta, cost] = fmincg(@(t)linearRegCostFunction(t, X, y, lambda), Theta, options);

plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
hold on;
pred_y = [ones(numel(X),1) X] * theta;
plot(X, pred_y, '-', 'LineWidth', 2);
hold off;

%Learning Curve Function
function learningCurve(inital_theta, X, y, val_X, val_y, lambda)
    m=size(X,1)
    J_val = zeros(m,1);
    J_train = zeros(m,1);
    for i = 1:size(X,1)
        train_X = X(1:i,:);
        train_y = y(1:i,:);
        options = optimset('MaxIter', 200, 'GradObj', 'On');
        [theta, cost] = fmincg(@(t)linearRegCostFunction(t, train_X, train_y, lambda), inital_theta, options);
        J_i_train = cost(end)
        J_i_val = linearRegCostFunction(theta, val_X, val_y, lambda);
        J_val(i) = J_i_val;
        J_train(i,:) = J_i_train;
    end
    plot(1:m, J_train, 1:m, J_val);
    title('Learning curve for linear regression')
    legend('Training', 'Cross Validation')
    xlabel('number of training examples')
    ylabel('Cost of model')
end

learningCurve(Theta, X, y, Xval, yval,0);

%Poly features function
function poly_X = polyfeatures(X, p)
    poly_X = zeros(numel(X), p)
    for i = 1:p
        poly_X(:,i) = X .^ i 
    end
end

%features normalized function
function [X_norm, mu, std] = featurenormal(X)
    mu = mean(X);
    std = std(X);
    X_norm = (X - mu) ./ std;
end

poly_X = polyfeatures(X, 8);
[X_norm,mu,std] = featurenormal(poly_X);

Theta = zeros(size(X_norm, 2) + 1, 1);

lambda = 0;
options = optimset('MaxIter', 200,'GradObj','On');
[theta, cost] = fmincg(@(t)linearRegCostFunction(t,  X_norm, y, lambda), Theta, options);
[ones(size(X_norm,1),1) X_norm]*theta

plot(X, y, 'rx', 'MarkerSize', 10, 'LineWidth', 1.5);
hold on;
X_plot = [min(X)-15:0.05:max(X)+25]';
X_plot_poly = polyfeatures(X_plot, 8);
X_plot_norm = (X_plot_poly - mu) ./ std
y_plot = [ones(size(X_plot_norm,1),1) X_plot_norm] * theta;
plot(X_plot, y_plot, '--', 'LineWidth', 2);

























