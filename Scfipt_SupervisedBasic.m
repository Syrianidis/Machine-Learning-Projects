% Script_5 Askhsh 5

close all;

% we add the PRTools directory
addpath('C:\Users\Michael\Desktop\Matlab\PRTools')

% randomPermutation returns two datasets of 212 samples with 30 attributes
%   each
[mal,ben]=randomPermutation;

% we keep only the first four attributes
mal=mal(:,1:4);
ben=ben(:,1:4);

% we normalize for each attribute - column
for j=1:4
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
set=[mal(vec(end/2+1:end),:);ben(vec(end/2+1:end),:)];

% we compute mean estimations and covariance matrices for each training set
mean_tr_mal=sum(train_mal)/size(train_mal,1);
mean_tr_ben=sum(train_ben)/size(train_ben,1);
s_tr_mal=(train_mal-ones(size(train_mal,1),1)*mean_tr_mal)'*(train_mal-ones(size(train_mal,1),1)*mean_tr_mal)/size(train_mal,1);
s_tr_ben=(train_ben-ones(size(train_ben,1),1)*mean_tr_ben)'*(train_ben-ones(size(train_ben,1),1)*mean_tr_ben)/size(train_ben,1);

% we pre-allocate
G_pdf_mal=zeros(1,numel(vec));
G_pdf_ben=zeros(1,numel(vec));

% we create the probability density fuction for our test set based on the
%   training sets
for i=1:size(set,1)
    G_pdf_mal(i)=exp(-.5*(set(i,:)-mean_tr_mal)*(s_tr_mal\(set(i,:)-mean_tr_mal)'))/sqrt((2*pi)^rank(s_tr_mal)*det(s_tr_mal));
    G_pdf_ben(i)=exp(-.5*(set(i,:)-mean_tr_ben)*(s_tr_ben\(set(i,:)-mean_tr_ben)'))/sqrt((2*pi)^rank(s_tr_ben)*det(s_tr_ben));
end

Gmal=sum(exp(-.5*(set-ones(length(set),1)*mean_tr_mal)*(s_tr_mal\(set-ones(length(set),1)*mean_tr_mal)'))/sqrt((2*pi)^rank(s_tr_mal)*det(s_tr_mal)))/length(set);
Gben=sum(exp(-.5*(set-ones(length(set),1)*mean_tr_ben)*(s_tr_ben\(set-ones(length(set),1)*mean_tr_ben)'))/sqrt((2*pi)^rank(s_tr_ben)*det(s_tr_ben)))/length(set);

% we predict in which class each sample will be classified
%   1 is the malignant class, 2 is the benign class
% we also say that malignant is the positive class
%   and benign is the negative class
act_class=ones(1,size(set,1));
act_class(end/2+1:end)=2;

% and we classify based on the gaussian pdf
pred_class=ones(1,size(set,1));
pred_class(G_pdf_mal<=G_pdf_ben)=2;

% we now compare for each sample if the predicted class is the actual class
y=pred_class-act_class;

% true positive - hit - for malignant instances
p=y(1:end/2);
TP=numel(p(p==0));

% true negative - correct rejection - for malignant instances
q=y(end/2+1:end);
TN=numel(q(q==0));

% false alarm and miss for malignant instances
FP=numel(y(y>0));
FN=numel(y(y<0));

% accuracy of classification
accuracy=(TP+TN)/numel(y);

% True Positive Rate and True Negative Rate for malignant instances
sensitivity=TP/(TP+FN);
specificity=TN/(FP+TN);

fprintf('Accuracy: %d percent\nTPR: %d percent\n\n',int8(100*accuracy),int8(100*sensitivity))