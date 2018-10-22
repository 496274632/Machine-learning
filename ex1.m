%% homework of Machine learning online class by Andrew Ng
%  Excise 1 Linear Regression with multiple variable 
			

%% Initialization;
clear; close all; clc;


%===================================Part 1 :Basic Function =======================================
fprintf('Running warmUpExercise ... \n' );
fprintf('General a 5*5 identity matrix: \n');
A=eye(5)

%===================================Part 2 :Plotting =============================================

data=load('/Users/Jianfeng/learning/Machine learning/machine-learning-ex1/ex1/ex1data1.txt');
X=data(:,1);
y=data(:,2);
m=length(y);
plot(X,y,'rx','MarkerSize',10);
ylabel('Profit in $10,000s');
xlabel('Population of city in 10,000s');

%===================================Part 2.2.3 :Computing the Cost Function=========================
X=[ones(m,1), data(:,1)];
theta=zeros(2,1);
J=sum(sum((X*theta-y).^2))/m/2;


%===================================Part 2.2.4 :Gradient decent====================================
alpha=0.01;
iteration=1500;
J_history=zeros(iteration,4);
for i=1:iteration
	temp1=(X'*(X*theta-y))./m
	temp0=sum((X*theta-y))./m
	temp1(1,:)=temp0
	theta=theta-temp1.*alpha
	J_history(i,:)=[i sum(sum((X*theta-y).^2))/m/2 theta']
end;
%''
plot(X(:,2),y,'rx','MarkerSize',10);
ylabel('Profit in $10,000s');
xlabel('Population of city in 10,000s');
hold on;
plot(X(:,2),X*theta,'-');
legend('Training data','Linear Regression');
hold off; 
plot(J_history(:,1),J_history(:,2),'-');
xlabel('numbers of iterations');
ylabel('Cost function of Linear Regression');

%%% Surface plot

theta0_vals = linspace(-10, 10, 100);
theta1_vals = linspace(-1, 4, 100);
J_2d=zeros(100,100);
for i=1:100
	for j=1:100
	t=[theta0_vals(i);theta1_vals(j)];
	J_2d(i,j)=sum((X*t-y).^2)/m/2;
	end
end	

figure;
J_2d=J_2d';
surf(theta0_vals,theta1_vals,J_2d);
xlabel('\theta_0');ylabel('\theta_1');

%'
%%% Contour plot
figure;
contour(theta0_vals, theta1_vals, J_2d, logspace(-2, 3, 20))
hold on;
plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);


%%%===========================================Optional Exercise======================================
data=load('/Users/Jianfeng/learning/Machine learning/machine-learning-ex1/ex1/ex1data2.txt');
X=data(:,1:2);
y=data(:,3);
%feature normalize
X_norm=(X-mean(X))./std(X);
m=size(X_norm,1);
X=[ones(m,1),X_norm]
feature_num=size(X,2);
theta=zeros(feature_num,1);

%%% Cost function
J=sum((X*theta-y).^2)/2/m;

%%%Gradient decent
num_iters=50;
alpha=[0.001 0.003 0.01 0.03 0.1 0.3]';
color=['r';'b';'g';'c';'y';'k']
J_history=zeros(num_iters,feature_num+2)
for ii=1:length(alpha)
	theta=zeros(feature_num,1)
for i=1:num_iters
	theta_grad=X'*(X*theta-y)/m
    theta_grad(1)=sum(X*theta-y)/m
    theta=theta-alpha(ii)*theta_grad
    J=sum((X*theta-y).^2)/2/m
    J_history(i,:)=[i theta' J]
end
plot(J_history(:,1),J_history(:,5),'color',color(ii),'LineWidth',2);
hold on;
end
legend("alpha=0.001","alpha=0.003","alpha=0.01","alpha=0.03","alpha=0.1","alpha=0.3");

%'
%%%=====================================Normal equations=========================================
theta_normal=pinv(X'*X)*X'*y;
predict_X=[1650 3];
predict_X_normal=(predict_X-mean(data(:,1:2)))./std(data(:,1:2));
pre_normal_X=[1 predict_X_normal];
predict_y=pre_normal_X*theta_normal;
predict_y_grad=pre_normal_X*theta;




%%%End===========================================================================================














































