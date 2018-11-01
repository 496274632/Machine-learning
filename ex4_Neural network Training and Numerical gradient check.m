%================================Exercise 4. Neural network Learning=========================
%displayData Function
function [h,display_array]=displayData(X,example_width)

if ~exist('example_width','var') || isempty(example_width)
    example_width = round(sqrt(size(X,2)));
end

colormap(gray);

[m,n] = size(X);
example_height = n / example_width;
display_rows = floor(sqrt(m));
display_cols = ceil(m / display_rows);
pad=1;
display_array = -ones(pad + display_rows * (example_width + pad),...
                      pad + display_cols * (example_height + pad));


curr_ex=1;
for i = 1:display_rows
    for j = 1:display_cols
        if curr_ex > m
            break;
        end
        max_val = max(abs(X(curr_ex,:)))

        display_array(pad + (i-1) * (example_height + pad) + (1 : example_height),...
                      pad + (j-1) * (example_width + pad) + (1 : example_width))...
        = reshape(X(curr_ex,:), example_height, example_width)/max_val

        curr_ex=curr_ex+1;
        end
    if curr_ex > m 
        break
    end
end

h = imagesc(display_array,[-1 1]);
axis image off;
drawnow;
end

load('/Users/jianfeng/learning/Machine learning/machine-learning-ex4/ex4/ex4data1.mat');
[m,n] = size(X);
sel = randperm(m);
sel = sel(1:100);
displayData(X(sel,:));

%=============================nnCostFunction=========================

%activation function
function g=sigmoid(z)
    g=1 ./ (1+e.^-z)
end

%sigmoidGradient function
function g = sigmoidGradient(z)
    g = z .* (1-z)
end

function [J,grad] = nnCostFunction(nn_params, input_layers, hidden_layers, num_labels,...
                            X, y, lambda)
    [m,n] = size(X)
    Theta1 = reshape(nn_params(1:hidden_layers * (input_layers + 1)),...
                    hidden_layers, input_layers + 1) 
    Theta2 = reshape(nn_params((hidden_layers * (input_layers + 1)) + 1 : end),...
                    num_labels, hidden_layers + 1)
    X = [ones(m,1) X];
    z2 = X * Theta1';
    a2 = sigmoid(z2);
    z3 = [ones(m,1) a2] * Theta2';
    a3 = sigmoid(z3);
    Theta_reg = [(Theta1(:,2:end))(:); (Theta2(:,2:end))(:)];
    for i = 1:num_labels
        yk(:,i)=(y==i)
    end;
    J=0;
    for i = 1:num_labels
        J += (-yk(:,i)'* log(a3(:,i)) - (1-yk(:,i))' * log (1 - a3(:,i))) / m
    end
    J = J + lambda * sum(Theta_reg.^2)/2/m

    delta3 = a3 - yk; %5000*10
    Theta2_grad = delta3' * [ones(m,1) a2]/m; 
    Theta2_grad(:,2:end) += lambda * Theta2(:,2:end) / m;
    delta2 = (delta3 * Theta2(:,2:end)) .* sigmoidGradient(a2); 
    Theta1_grad = delta2' *  X / m; 
    Theta1_grad(:,2:end) += lambda * Theta1(:,2:end) / m
    grad = [Theta1_grad(:); Theta2_grad(:)]
end

load('/Users/jianfeng/learning/Machine learning/machine-learning-ex4/ex4/ex4weights.mat');
nn_params = [Theta1(:); Theta2(:)];
input_layers = 400;
hidden_layers = 25;
num_labels = 10;
lambda = 3;
[J, grad] = nnCostFunction(nn_params, input_layers, hidden_layers, num_labels,...
                            X, y, lambda)


%==========================Check costgradient function=========================

%%% debug_initalize_weight function
function W = debug_initalize_weight(fan_out, fan_in )
    W = zeros(fan_out, fan_in + 1);
    W = reshape(sin(1:numel(W)), size(W)) / 10;
end 

%%% computeNumericalGradient function
function numgrad = computeNumericalGradient(J, theta)
    numgrad = zeros(numel(theta),1);
    pergrad = zeros(numel(theta),1);
    e = 1e -4
    for i = 1:numel(theta)
        pergrad(i) = e
        theta_add = theta + pergrad
        theta_loss =theta - pergrad
        J_loss = J(theta_loss)
        J_add = J(theta_add)
        numgrad(i) = (J_add - J_loss)/(2 * e)
        pergrad(i) = 0
    end
end

%%% Numeric Check cost function
function checkNNGradients(lambda)
    if ~exist('lambda', 'var') || isempty(lambda)
        lambda = 0
    end
    input_layers = 3;
    hidden_layers = 5;
    num_labels = 3;
    m = 5;
    Theta1 = debug_initalize_weight(hidden_layers, input_layers );
    Theta2 = debug_initalize_weight(num_labels, hidden_layers);
    nn_params = [Theta1(:);Theta2(:)]
    X = debug_initalize_weight(m, input_layers - 1);
    y = 1 + mod(1:m, num_labels);
    costFunc = @(p)nnCostFunction(p,input_layers, hidden_layers, num_labels,...
                 X, y, lambda);
    [cost, grad] = costFunc(nn_params);
    numgrad = computeNumericalGradient(costFunc, nn_params);
    disp([numgrad grad]);
    fprintf(['The above two columns you get should be very similar.\n' ...
             '(Left-Your Numerical Gradient, Right-Analytical Gradient)\n\n']);
    diff = norm(numgrad-grad)/norm(numgrad+grad);

    fprintf(['If your backpropagation implementation is correct, then \n' ...
             'the relative difference will be small (less than 1e-9). \n' ...
             '\nRelative Difference: %g\n'], diff);
end

%%% Check nnCostFunction and Gradient of theta
checkNNGradients;

lambda=3;
checkNNGradients(lambda);


%=====================================Training Neural network====================================

%%%'Check
function p = predict (Theta1, Theta2, X)
    [m,n]=size(X);
    a1 = [ones(m,1) X];
    z2 = a1 * Theta1';
    a2 = sigmoid(z2);
    z3 = [ones(m,1) a2] * Theta2';
    a3 = sigmoid(z3);
    [temp,p] = max(a3, [], 2 );
    p(p==10)=0;
end
options = optimset('MaxIter', 50);
[nn_params, cost] = fminunc(costFunc, nn_params, options);
Theta1 = reshape(nn_params(1:hidden_layers * (input_layers + 1)),...
                hidden_layers, input_layers + 1);
Theta2 = reshape(nn_params(numel(Theta1) + 1:end), num_labels, hidden_layers + 1);

rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(X(rp(i), :));

    pred = predict(Theta1, Theta2, X(rp(i),:));
    fprintf('\nNeural Network Prediction: %d ( %d)\n', pred);
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end


lambda_vector=[0.01 0.03 0.1 0.3]';
J=zeros(numel(lambda_vector),1);
for i=1:length(lambda_vector)
    lambda = lambda_vector(i)
    [nn_params, cost] = fminunc(@(t)nnCostFunction(t,input_layers, hidden_layers, num_labels,...
                 X, y, lambda), nn_params, options);
    J(i) = cost
end
plot(lambda_vector, J, 'k+','LineWidth',2)

    













