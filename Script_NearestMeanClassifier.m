% Script_15

close all;clear all;clc

addpath('C:\Users\Michael\Desktop\Matlab\PRTools')

k=2.^(1:4);

[data_mal,data_ben] = randomPermutation();

% temporary accuracy [before mean computation]
accuracy_forward = zeros(1,length(k)+1); 
accuracy_l2_r1=zeros(1,length(k)+1); % adding extra columh for noc = 29
accuracy_l3_r2=zeros(1,length(k)+1); % adding extra columh for noc = 29
accuracy_l4_r3=zeros(1,length(k)+1); % adding extra columh for noc = 29

% for i = 1:10 % cross validation [10 folds]
vec = randperm(212); %permutate indexes
train=vertcat(data_mal(vec(1:end/2),:),data_ben(vec(1:end/2),:)); %training set
test=vertcat(data_mal(vec(end/2+1:end),:),data_ben(vec(end/2+1:end),:)); % test set

act_class=ones(size(test,1),1);
act_class(end/2+1:end)=2;
% add training set into dataset & assign classes
train_set=dataset(train,act_class);
test_set=dataset(test,act_class);

for i=1:length(k) % number of characteristics
%     
    map=featselm(train_set,'maha-s','forward',k(i));
    train_c=train_set*map; % create pruned training set
    map_c=nmc(train_c); % create map_c mapping
    pred_class=test_set*map*map_c*labeld; % classify test_set
    
    TP=numel(pred_class(pred_class(1:end/2)==1));% count true positives
    TN=numel(pred_class(pred_class(end/2+1:end)==2)); % count true negatives
    accuracy_forward(i)=(TP+TN)/size(pred_class,1);
    if i==length(k)
        accuracy_forward(end)=(TP+TN)/size(test,1);
    end
%     
    map=featsellr(train_set,'maha-s',k(i),2,1);
    train_c=train_set*map; % create pruned training set
    map_c=nmc(train_c); % create map_c mapping
    pred_class=test_set*map*map_c*labeld; % classify test_set
    
    TP=numel(pred_class(pred_class(1:end/2)==1));% count true positives
    TN=numel(pred_class(pred_class(end/2+1:end)==2)); % count true negatives
    accuracy_l2_r1(i)=(TP+TN)/size(test,1);
    if i==length(k)
        accuracy_l2_r1(end)=(TP+TN)/size(test,1);
    end
%     
    map=featsellr(train_set,'maha-s',k(i),3,2);
    train_c=train_set*map; % create pruned training set
    map_c=nmc(train_c); % create map_c mapping
    pred_class=test_set*map*map_c*labeld; % classify test_set
    
    TP=numel(pred_class(pred_class(1:end/2)==1));% count true positives
    TN=numel(pred_class(pred_class(end/2+1:end)==2)); % count true negatives
    accuracy_l3_r2(i)=(TP+TN)/size(test,1);
    if i==length(k)
        accuracy_l3_r2(end)=(TP+TN)/size(test,1);
    end
%     
    map=featsellr(train_set,'maha-s',k(i),4,3);
    train_c=train_set*map; % create pruned training set
    map_c=nmc(train_c); % create map_c mapping
    pred_class=test_set*map*map_c*labeld; % classify test_set
    
    TP=numel(pred_class(pred_class(1:end/2)==1));% count true positives
    TN=numel(pred_class(pred_class(end/2+1:end)==2)); % count true negatives
    accuracy_l4_r3(i)=(TP+TN)/size(test,1);
    if i==length(k)
        accuracy_l4_r3(end)=(TP+TN)/size(test,1);
    end
end

% visualize results
plot([k 2^(length(k)+1)],accuracy_forward,'c-*',[k 2^(length(k)+1)],accuracy_l2_r1,'b-s',[k 2^(length(k)+1)],accuracy_l3_r2,'r-o',[k 2^(length(k)+1)],accuracy_l4_r3,'g-p');
legend('forward','l=2, r=1','l=3, r=2','l=4, r=3'),grid on,xlabel('k = 2   -   4'),ylabel('accuracy')
title('Forward Selection \newlineP-L-Takeaway-R Selection\newlineWithout 10-fold Cross Validation'),axis([1 9 0.75 .95])
xlabel('Number of Characteristics'),ylabel('Accuracy')