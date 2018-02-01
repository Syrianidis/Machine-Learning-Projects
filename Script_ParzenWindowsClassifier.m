% Script_13 Askhsh 13 Decision Theory

% close all;clear all;clc

addpath('C:\Users\Michael\Desktop\Matlab\PRTools')

% randomPermutation returns two datasets of 212 samples with 30 features
%   each
[mal,ben]=randomPermutation;

% we keep only the first four features
mal=mal(:,1:4);
ben=ben(:,1:4);

% we normalize for each feature - column
for j=1:size(mal,2)
    mal(:,j)=mal(:,j)-min(mal(:,j));
    mal(:,j)=mal(:,j)/max(mal(:,j));
    ben(:,j)=ben(:,j)-min(ben(:,j));
    ben(:,j)=ben(:,j)/max(ben(:,j));
end

% we use the random vector as index to pick our training sets
vec=randperm(size(mal,1));
train_mal=mal(vec(1:end/2),:);
train_ben=ben(vec(1:end/2),:);
% the rest are now the test set that we will classify
test_mal=mal(vec(end/2+1:end),:);
test_ben=ben(vec(end/2+1:end),:);
set=vertcat(test_mal,test_ben);

% again we initialize all we need
% the dimensionality
rnk=size(mal,2);
% the bandwidths we are to use on the gaussian kernel pdf
h=.01:.01:1;
% the space for conditional probabilities (pdf) of each class
p_mal=zeros(numel(h),size(set,1));
p_ben=zeros(numel(h),size(set,1));
% we iterate for each bandwidth
for j=1:numel(h)
%     we iterate for each instance of test set
    for i=1:size(set,1)
%         we use this for easily avoiding errors
        x=train_mal-ones(size(train_mal,1),1)*set(i,:);
%         we iterate for each instance of malignant train sets
        for k=1:size(x,1)
%             we compute the pdf of malignant set
            p_mal(j,i)=p_mal(j,i)+exp(-.5*x(k,:)*(x(k,:)')/h(j)^2)/((2*pi)^(rnk/2)*h(j)^rnk);
        end
    end
%     and we divide by the number of instances of test set, in order to
%       compute the means of each instance's pdf of test set
    p_mal(j,:)=p_mal(j,:)/i;
%     same things as in lines [42,54] just for the benign class
    for i=1:size(set,1)
        x=train_ben-ones(size(train_ben,1),1)*set(i,:);
        for k=1:size(x,1)
            p_ben(j,i)=p_ben(j,i)+exp(-.5*x(k,:)*(x(k,:)')/h(j)^2)/((2*pi)^(rnk/2)*h(j)^rnk);
        end
    end
    p_ben(j,:)=p_ben(j,:)/i;
end

% we pre-allocate the space for accuracy
accuracy=zeros(size(h));
% and we initialize the expected classification of malignant and benign
%   instances from test set, in which 1 corresponds to class malignant, 2
%   corresponds to class benign
act_class=ones(1,size(set,1));
act_class(end/2+1:end)=2*act_class(end/2+1:end);

% we classify all instances in class 1
pred_class=ones(1,size(set,1));
% then for each bandwidth we correct the classification for class 2
for i=1:numel(h)
% if the conditional probability for class 2 is greater than the
%   probability of class 1, then the instance is assigned to class 2
    pred_class(p_mal(i,:)<p_ben(i,:))=2;
% we now compare for each sample if the predicted class is the actual class
    y=pred_class-act_class;
% true positive - hit - for malignant instances
    p=y(1:end/2);
    TP=numel(p(p==0));
% true negative - correct rejection - for malignant instances
    q=y(end/2+1:end);
    TN=numel(q(q==0));
% accuracy of classification
    accuracy(i)=(TP+TN)/numel(y);
end

% this section is for visualizing the accuracy according to the bandwidth
% plot(h,accuracy),axis([.9*min(h) 1.1*max(h) .9*min(accuracy) 1.1*max(accuracy)])
% title('Accuracy of Kernel Density Estimation'),xlabel('Bandwidth "h"'),ylabel('Accuracy')