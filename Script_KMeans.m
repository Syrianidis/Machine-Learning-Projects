% Script_14 Askhsh 14 Decision Theory

% close all;clear all;clc

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
test_set=vertcat(test_mal,test_ben);

% we compute the accuracy for Nearest Means methods
[~,accuracy_eu_nm]=nearestmean(train_mal,train_ben,test_set,'euclidean');
[~,accuracy_ma_nm]=nearestmean(train_mal,train_ben,test_set,'mahalanobis');

% and the accuracies for different values of k
k=1:212;
accuracy_knn=zeros(size(k));
for i=1:max(k)
    [~,accuracy_knn(i)]=knearestneighbor(train_mal,train_ben,test_set,i);
end

% q=[accuracy_knn accuracy_eu_nm accuracy_ma_nm];
% plot(k,accuracy_knn,k,accuracy_eu_nm,k,accuracy_ma_nm)
% title('Accuracy for K-Nearest Neighbors VS Nearest Means')
% xlabel('Values for K'),ylabel('Accuracy'),legend('KNN accuracy','NM with Euclidean','NM with Mahalanobis')
% axis([.9 max(k) .9*min(q) 1.1*max(q)])
% 
% k_max=k(accuracy_knn==max(accuracy_knn));