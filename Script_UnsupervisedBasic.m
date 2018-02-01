% Script_12 Askhsh 12 Decision Theory

close;clear;clc

% (a)

% we initialize all we are to use
% we initialize the means we will use to generate datasets X1, X2
m1_a=[4 4 3];
m2_a=[2 2 1];
% we initialize the covariance matrix for datasets X1, X2
S=.8*eye(3);

% we generate datasets with 500 3-D instances X1, X2
x1=mvnrnd(m1_a',S,500);
x2=mvnrnd(m2_a',S,500);

% (b)

% we compute the means of each dataset with ML (Maximum Likelyhood)
m1_b=sum(x1)/size(x1,1);
m2_b=sum(x2)/size(x2,1);

% we compute the covariance matrices of each dataset with ML
S1=(x1-ones(length(x1),1)*m1_b)'*(x1-ones(length(x1),1)*m1_b)/length(x1);
S2=(x2-ones(length(x2),1)*m2_b)'*(x2-ones(length(x2),1)*m2_b)/length(x2);

% (c)

% we initialize the instance we are to classify in dataset X1 or X2
x=[2 3 4];

% we compute X's euclidean distance for each dataset
EuDi1=(x-m1_b)*(x-m1_b)';
EuDi2=(x-m2_a)*(x-m2_a)';

% and we classify the instance to a dataset depending of it's distance
% if it's closer to X1, then it's classified in X1
if EuDi1<EuDi2
    fprintf('Based on the Euclidean distance X is classified in X1\n')
%     if it's closer to X2, then it's classified to X2
elseif EuDi1>EuDi2
    fprintf('Based on the Euclidean distance X is classified in X2\n')
%     if it's exactly at the middle, it's classified randomly, although we
%       should dispose of it and test a different instance
else
    if rand(1)>.5
        fprintf('Based on the Euclidean distance X is classified in X1\n')
    else
        fprintf('Based on the Euclidean distance X is classified in X2\n')
    end
end

% (d)

% we compute X's mahalanobis distance from datasets X1, X2
MaDi1=(x-m1_b)*(S1\(x-m1_b)');
MaDi2=(x-m2_a)*(S\(x-m2_a)');
% and we repeat the same steps as in lines [37,52]
if MaDi1<MaDi2
    fprintf('Based on the Mahalanobis distance X is classified in X1\n')
elseif MaDi1>MaDi2
    fprintf('Based on the Mahalanobis distance X is classified in X2\n')
else
    if rand(1)>.5
        fprintf('Based on the Mahalanobis distance X is classified in X1\n')
    else
        fprintf('Based on the Mahalanobis distance X is classified in X2\n')
    end
end

% (e)

% we initialize all we are to use
% the probability of class 1 (dataset X1)
p1=.4;
% the probability of class 2 (dataset X2)
p2=1-p1;
% we call function GAUSS() to compute the pdf of each dataset
p_cond_1=Gauss(m1_b,S1,x1,3);
p_cond_2=Gauss(m2_a,S,x2,3);
% we compute the a-posteriori probabilities for each dataset
a_post_1=p_cond_1*p1;
a_post_2=p_cond_2*p2;
% we classify instance X based on the probabilities of Bayes' theorem
if a_post_1>a_post_2
    fprintf('Based on the Bayesian Theorem X is classified in X1\n')
elseif a_post_1<a_post_2
    fprintf('Based on the Bayesian Theorem X is classified in X2\n')
else
    if rand(1)>.5
        fprintf('Based on the Bayesian Theorem X is classified in X1\n')
    else
        fprintf('Based on the Bayesian Theorem X is classified in X2\n')
    end
end

% (f)

% we initialize all we are to use
% we initialize the means of the datasets we are to generate
m1_b=.5*ones(1,2);
m2_b=3.5*ones(1,2);
% and the values we are to give to the covariance matrices
z=[1 1 1 1
    .8 .5 .3 .8
    .5 .5 0 .5
    2 2 0 .1];
% we itarate 4 times, each one for every covariance matrix that is given
for i=1:size(z,1)
%     we initiazlie the covariance matrices for each dateset
    S1=[z(i,1)^2 z(i,3);z(i,3) z(i,2)^2];
    S2=z(i,4)*eye(2);
%     we generate the datasets based on the means and covariance matrices
%       we initialized
    X1=mvnrnd(m1_b,S1,600);
    X2=mvnrnd(m2_b,S2,600);
%     then we plot each dataset based on their features
    figure()
    plot(X1(:,1),X1(:,2),'bx',X2(:,1),X2(:,2),'r+')
    title(['S1=',num2str(S1(1)),', ',num2str(S1(3)),'; ','S2=',num2str(S2(1)),', ',num2str(S2(3)),'\newline','      ',num2str(S1(2)),', ',num2str(S1(4)),'         ',num2str(S2(2)),', ',num2str(S2(4))])
    legend('Dataset X1','Dataset X2'),xlabel('Feature 1'),ylabel('Feature 2')
end