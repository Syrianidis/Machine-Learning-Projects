function [pred_class,accuracy]=knearestneighbor(train_mal,train_ben,test_set,k)

% we predict in which class each sample will be classified
%   1 is the malignant class, 2 is the benign class
act_class=ones(1,size(test_set,1));
act_class(end/2+1:end)=2;
% we initialize the instances as if they were all classified in class 1 and
%   then we will correct it with the following computations
pred_class=ones(1,size(test_set,1));
% we check for every instance if it was correctly classified
for i=1:size(test_set,1)
%     we compute the euclidean distance from malignant training set
    eudi_mal=zeros(size(1:size(train_mal,1)));
    for j=1:size(train_mal,1)
        eudi_mal(j)=sqrt((test_set(i,:)-train_mal(j,:))*(test_set(i,:)-train_mal(j))');
    end
%     we compute the euclidean distance from benign training set
    eudi_ben=zeros(size(1:size(train_ben,1)));
    for j=1:size(train_ben,1)
        eudi_ben(j)=sqrt((test_set(i,:)-train_ben(j,:))*(test_set(i,:)-train_ben(j))');
    end
%     we concatenate them in order to find which ones are lower
    distances=horzcat(eudi_mal,eudi_ben);
%     then we sort them in ascending order in order to gather the lower
%       ones close, but we only need their positions, so index will give as
%       the nearest neighbors to each instance
    [~,q]=sort(distances);
%     this section is for deciding in which class we will classify each
%       instance
    p=zeros(length(distances)/2,2);
    p(q(1:k))=1;
    p=sum(p);
%     if more indexes corresponded to class 2, then we correct the
%       classification
    if p(2)>p(1)
        pred_class(i)=2;
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