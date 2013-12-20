function classhypo = KNN(features_train,features_test,labels_train,k,weighted)
% function classhypo = KNN(features_train,features_test,labels_train,k,weighted)
% 
% Performs KNN classification to a set of data points using a given
% training data and the associated class labels.
%
%   Inputs:
%       features_train:      N x F matrix containing the training data, 
%                               where N is the number of samples
%                               and F is the number of features.
%       features_test:       M x F matrix containing the classification data, 
%                               where M is the number of samples
%                               and F is the number of features.
%       labels_train:        N x 1 vector of class labels for training
%                               samples
%       k:                   number of nearest neighbors used in the
%                               classification
%       weighted:            weight nearest neighbor counts by the inverse of 
%                               class frequencies estiamted from the training
%                               data? (0/1, default: 1).

persistent pdist2_exists;

if(size(features_train,1) ~= length(labels_train))
    error('Different number of training samples and class labels');
end

if ~exist('weighted','var') || isempty(weighted)
  weighted = 1;
end

% Compute the distribution of samples per each class in the training data

if(weighted == 1)
    N_classes = max(labels_train);
    w = zeros(N_classes,1);
    for c = 1:N_classes
        w(c) = sum(labels_train == c);
    end
end

if isempty(pdist2_exists)
    pdist2_exists = exist('pdist2','file');
end

if pdist2_exists
    D = pdist2(features_test,features_train,'euclidean');    % Compute distances between train and dev vectors
else
    D = zeros(size(features_test,1),size(features_train,1));
    for j=1:size(features_test,1)
        D(j,:) = sqrt(sum((repmat(features_test(j,:),size(features_train,1),1)-features_train).^2,2))';
    end
end

[tmp,orderi] = sort(D,2,'ascend');                     % Sort distances into ascending order

% Get classes of k nearest neighbors for each dev sample
nearest =  labels_train(orderi(:,1:k));

% Get class hypothesis for each sample by (weighted) majority voting
classhypo = zeros(size(nearest,1),1);
for j = 1:size(nearest,1);
    a = zeros(N_classes,1);
    for c = 1:N_classes
        if(weighted == 1)
            a(c) = sum(nearest(j,:) == c)./w(c); % weighting with training data distribution "w".
        else
            a(c) = sum(nearest(j,:) == c); 
        end
    end
    [tmp,classhypo(j)] = max(a);
end
