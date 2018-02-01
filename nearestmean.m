function [pred_class,accuracy]=nearestmean(train_mal,train_ben,test_set,distance_type)

% we compute the means of each dataset
mean_mal=mean(train_mal);
mean_ben=mean(train_ben);
% we predict in which class each sample will be classified
%   1 is the malignant class, 2 is the benign class
act_class=ones(1,size(test_set,1));
act_class(end/2+1:end)=2;
% we initialize the instances as if they were all classified in class 1 and
%   then we will correct it with the following computations
pred_class=ones(1,length(test_set));
% depending on the type of distance we are to use we make different
%   computations
if strcmp(distance_type,'euclidean')
%     we compute the distances for all instances of test_set
    for i=1:size(test_set,1)
        eudi_mal=sqrt((test_set(i,:)-mean_mal)*(test_set(i,:)-mean_mal)');
        eudi_ben=sqrt((test_set(i,:)-mean_ben)*(test_set(i,:)-mean_ben)');
%         if we were wrong in the first place, we correct it
        if eudi_mal>eudi_ben
            pred_class(i)=2;
        end
    end
%     same as in lines [15,24]
elseif strcmp(distance_type,'mahalanobis')
    S_mal=(train_mal-ones(length(train_mal),1)*mean_mal)'*(train_mal-ones(length(train_mal),1)*mean_mal);
    S_ben=(train_ben-ones(length(train_ben),1)*mean_ben)'*(train_ben-ones(length(train_ben),1)*mean_ben);
    for i=1:size(test_set,1)
        madi_mal=sqrt((test_set(i,:)-mean_mal)*(S_mal\(test_set(i,:)-mean_mal)'));
        madi_ben=sqrt((test_set(i,:)-mean_ben)*(S_ben\(test_set(i,:)-mean_ben)'));
        if madi_mal>madi_ben
            pred_class(i)=2;
        end
    end
end
% we now compare for each sample if the predicted class is the actual class
y=pred_class-act_class;
% true positive - hit - for malignant instances
p=y(1:end/2);
TP=numel(p(p==0));
% true negative - correct rejection - for malignant instances
q=y(end/2+1:end);
TN=numel(q(q==0));
% accuracy of classification
accuracy=(TP+TN)/numel(y);