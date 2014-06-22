function [classhypo,D] = KNNI(features_train,features_test,labels_train,k,weighted,D0)
% function classhypo = KNNI(features_train,features_test,labels_train,k,weighted,D0)
% 
% k-nearest-neighbors classification with incremental update of distance matrix:
% performs KNN classification to a set of data points using a given
% training data and the associated class labels.
%
%   Inputs:
%       features_train:      N x F matrix containing the training data, 
%                               where N is the number of samples
%                               and F is the number of features.
%       features_test:       M x F matrix containing the classification data, 
%                               where M is the number of samples
%                               and F is the number of features.
%       labels_train:        N x 1 vector of integer class labels for training
%                               samples
%       k:                   number of nearest neighbors used in the
%                               classification
%       weighted:            weight nearest neighbor counts by the inverse of 
%                               class frequencies estiamted from the training
%                               data? (0/1, default: 1).
%       D0:                  previous distance matrix of squared Euclidean
%                            distances, can be used to speed up the
%                            computation of nested feature sets (M x N
%                            matrix or 0 to omit, default: 0)
%
%  Outputs:
%       classhypo:           predicted class labels
%       D:                   updated M x N distance matrix
%
%  Authors: Okko Rasanen, Jouni Pohjalainen, June 2014

persistent pdist2_exists;

N_train = size(features_train,1); % number of training samples
N_test  = size(features_test,1);  % number of test samples

if nargin<6 || (any(size(D0)~=[N_test,N_train]) && ~isscalar(D0))
    incremental = false;
else
    incremental = true;
end

if(N_train ~= length(labels_train))
    error('Different number of training samples and class labels');
end

if ~exist('weighted','var') || isempty(weighted)
  weighted = 1;
end

% Compute the distribution of samples per each class in the training data
N_classes = max(labels_train);
if weighted
    w = zeros(N_classes,1);
    for c = 1:N_classes
        w(c) = sum(labels_train == c);
    end
end

% Check if pdist2 function exists for distance computations (faster but 
% absent from older MATLAB versions). 
if isempty(pdist2_exists)
    pdist2_exists = exist('pdist2','file');
end

if incremental
    if pdist2_exists
        D = D0 + pdist2(features_test,features_train,'euclidean').^2;    % Compute distances between train and dev vectors
    else
        D = D0 + zeros(N_test,N_train);
        for j=1:N_test
            D(j,:) = D(j,:) + sum((repmat(features_test(j,:),N_train,1)-features_train).^2,2)';
        end
    end
else
    if pdist2_exists
        D = pdist2(features_test,features_train,'euclidean');    % Compute distances between train and dev vectors
    else
        D = zeros(N_test,N_train);
        for j=1:N_test
            D(j,:) = sqrt(sum((repmat(features_test(j,:),N_train,1)-features_train).^2,2))';
        end
    end
end

[tmp,orderi] = sort(D,2,'ascend');                     % Sort distances into ascending order

classhypo = zeros(N_test,length(k));
for i1=1:length(k)
    % Get classes of k nearest neighbors for each dev sample
    nearest =  labels_train(orderi(:,1:k(i1)));
    
    % Get class hypothesis for each sample by (weighted) majority voting

    a = zeros(N_test,N_classes);
    for c = 1:N_classes
        if weighted
            a(:,c) = sum(nearest==c,2)/w(c);
        else
            a(:,c) = sum(nearest==c,2);
        end
    end
    [tmp,classhypo(:,i1)] = max(a,[],2);
end
