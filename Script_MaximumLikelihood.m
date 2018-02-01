% Script_11 Askhsh 11 Decision Theory

close all;clear all;clc

% we initialize everything we need
% the mean
m=[2 -2]
% and the covariance matrix
S=[.9 .2;.2 .3];

% we generate a 2-D dataset
X=mvnrnd(m,S,1e2);

% we compute the dataset's mean
ML_x1=sum(X(:,1))/1e2;
ML_x2=sum(X(:,2))/1e2;
m_ML_1e2=[ML_x1 ML_x2];

% we could use the following code that was given, but it's slower and
%   harder to read than the code in line 33
% to be exact, it's 4~5 times slower
% 
% S=zeros(2);
% for i = 1:100,
% S(1,1) = S(1,1) + (X(i,1) - m_ML(1))*(X(i,1) - m_ML(1))';
% S(1,2) = S(1,2) + (X(i,1) - m_ML(1))*(X(i,2) - m_ML(2))';
% S(2,1) = S(2,1) + (X(i,2) - m_ML(2))*(X(i,1) - m_ML(1))';
% S(2,2) = S(2,2) + (X(i,2) - m_ML(2))*(X(i,2) - m_ML(2))';
% end
% S=S/1e2;

% we compute the dataset's covariance matrix
S_ML_1e2=(X-ones(100,1)*m_ML_1e2)'*(X-ones(100,1)*m_ML_1e2)/1e2;

% because the instances of the dataset we previously generated are not
%   enough for an accurate approximation, we generate a new dataset with
%   a lot more instances
X=mvnrnd(m,S,1e4);

% we compute the new dataset's mean
ML_x1=sum(X(:,1))/1e4;
ML_x2=sum(X(:,2))/1e4;
m_ML_1e4=[ML_x1 ML_x2];

% we compute the new dataset's covariance matrix
S_ML_1e4=(X-ones(1e4,1)*m_ML_1e4)'*(X-ones(1e4,1)*m_ML_1e4)/1e4;