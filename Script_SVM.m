% Script_16 Askhsh 16 Decision Theory

% close all;clear all;clc

% we add the PRTools directory
addpath('C:\Users\Michael\Desktop\Matlab\PRTools')

% randomPermutation returns two datasets of 212 samples with 30 attributes
%   each
[mal,ben]=randomPermutation;

% we use the random vector as index to pick our training set
vec=randperm(size(mal,1));
train_set=vertcat(mal(vec(1:end/2),:),ben(vec(1:end/2),:));
% the rest are now the test set that we will classify
test_set=vertcat(mal(vec(end/2+1:end),:),ben(vec(end/2+1:end),:));

% we predict in which class each sample will be classified
%   1 is the malignant class, 2 is the benign class
act_class=ones(size(test_set,1),1);
act_class(end/2+1:end)=2;

% we use SVM training and classification with the following Matlab commands
%   with linear kernel function
[svm_struct, svIndex]=svmtrain(train_set,act_class,'kernel_function','linear');
outclass=svmclassify(svm_struct,test_set);

% we locate which instances were classified correctly
y=outclass-act_class;
% true positive - hit - for malignant instances
p=y(1:end/2);
TP=numel(p(p==0));
% true negative - correct rejection - for malignant instances
q=y(end/2+1:end);
TN=numel(q(q==0));
% accuracy of classification
accuracy_linear=(TP+TN)/numel(y);

% we use SVM training and classification with the following Matlab commands
%   with quadratic kernel function
[svm_struct, svIndex]=svmtrain(train_set,act_class,'kernel_function','quadratic');
outclass=svmclassify(svm_struct,test_set);
% we locate which instances were classified correctly
y=outclass-act_class;
% true positive - hit - for malignant instances
p=y(1:end/2);
TP=numel(p(p==0));
% true negative - correct rejection - for malignant instances
q=y(end/2+1:end);
TN=numel(q(q==0));
% accuracy of classification
accuracy_quadratic=(TP+TN)/numel(y);

% we use SVM training and classification with the following Matlab commands
%   with radial basis kernel function
[svm_struct, svIndex]=svmtrain(train_set,act_class,'kernel_function','rbf');
outclass=svmclassify(svm_struct,test_set);
% we locate which instances were classified correctly
y=outclass-act_class;
% true positive - hit - for malignant instances
p=y(1:end/2);
TP=numel(p(p==0));
% true negative - correct rejection - for malignant instances
q=y(end/2+1:end);
TN=numel(q(q==0));
% accuracy of classification
accuracy_rbf=(TP+TN)/numel(y);