%===============================Logistic Regression=====================================

%====1.Data visulization=================
data=load('/Users/jianfeng/learning/Machine learning/machine-learning-ex2/ex2/ex2data1.txt');
data_X=data(:,1:2);
y=data(:,3);
X=data_X(y==1,:);
data_neg=data_X(y==0,:);
plot(X(:,2),data_pos(:,2),'k+','LineWidth',2,'MarkerSize',7);
hold on;
plot(data_neg(:,1),data_neg(:,2),'ko','MarkerFaceColor','y','MarkerSize',7);
legend('Admitted','Not Admitted');
hold off;

%====2.Function implement================
%%%%sigmoid function
function g=sigmoid(z)
	g=((1+e.^-z)).^-1
end

%%%%cost function
function [J,grad]=costFunction(theta,X,y,lambda)
	m=size(X,1);
	hx=sigmoid(X*theta);
	J=(-y'*log(hx)-(1-y)'*log(1-hx))/m+lambda*theta'*theta/2/m;
	grad=X'*(hx-y)/m+lambda*theta./m
	grad(1,:)=X(:,1)'*(hx-y)/m;
end

m=size(data_X,1);
theta_init=zeros(3,1);
X=[ones(m,1) data_X];
[J,grad]=costFunction(theta_init,X,y,0);

%'
%=======Constrained the cost function=====================
options=optimset('GradObj','on','MaxIter',400);
[theta,cost,exitFlag]=fminunc(@(t)costFunction(t,X,y),theta_init,options);

%=======Construct the plotDecisionBoundary function=====================================================

%=====first Construct the mapfeature function==========
function out=mapfeature(X1,X2,degree)
	out=ones(size(X1))
	for i=1:degree
      for j=0:i
        out(:,end+1)=(X1.^(i-j)).*(X2.^j)
	  end
	end
end

%=====Construct the plot function=======================
function plotDecisionBoundary(theta,X,y,degree)
X_pos=X(y==1,:)
X_neg=X(y==0,:)
plot(X_pos(:,2),X_pos(:,3),'k+','LineWidth',2,'MarkerSize',7);
hold on;
plot(X_neg(:,2),X_neg(:,3),'ko','MarkerFaceColor','y','MarkerSize',7);
if size(theta,1) <= 3
	plot_x=[min(X(:,2))-2,max(X(:,2))+2]
	plot_y=(theta(1)+theta(2).*plot_x)./(-theta(3))
	plot(plot_x,plot_y)
	legend('Admitted','Not Admitted','DecisionBoundary')
else
	u=linspace(-1,1.5,50)
	v=linspace(-1,1.5,50)
	z=zeros(length(u),length(v))
	%map feature to high demension space
	for i=1:length(u)
        for j=1:length(v)
            z(i,j)=mapfeature(u(i),v(j),degree)*theta;
        end
	end
	%transpose z to before calling contour
	z=z';
contour(u,v,z,[0,0],'LineWidth',2);
end
hold off;

end


%==================general multivariable logistic Regression DecisionBoundary plot=================='
data2=load('/Users/jianfeng/learning/Machine learning/machine-learning-ex2/ex2/ex2data2.txt');
X_multi=data2(:,1:2);
y_multi=data2(:,3);
X_multi_pos=X_multi(y_multi==1,:);
X_multi_neg=X_multi(y_multi==0,:);

%===================Data visulization and DecisionBoundary plot ========================

%map the feature to high demension space

X_map=mapfeature(X_multi(:,1),X_multi(:,2),6);
theta_init=zeros(size(X_map,2),1);
options=optimset('GradObj','on','MaxIter',400);
[theta,cost,exitFlag]=fminunc(@(t)costFunction(t,X_map,y_multi,1),theta_init,options);
X_multi_add=[ones(size(X_multi,1),1) X_multi];
plotDecisionBoundary(theta,X_multi_add,y_multi,6);








































